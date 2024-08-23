"""  Simulating the SP network
      used to create training dataset and for the main experiment with the NN """
from convert_data import generate_data
import os
import pickle
import argparse
import enum
import heapq
import math
import numpy as np
import gurobipy as gp
from scipy.spatial.distance import cdist
from collections import deque


# Define simulation events for the simulation heap queue
class EventType(enum.Enum):
    DEPOT_ARRIVAL = 1
    RECIPIENT_RESPONSE = 2
    DELIVERY = 3
    PARCEL_PICKUP = 4
    EXPECTED_FORECAST_UPDATE = 5


class AssignmentPolicy(enum.Enum):
    BL1 = "bl1"
    BL2 = "bl2"
    BL3 = "bl3"
    PREFERENCE = "preference"
    LOWER_BOUND = "lower_bound"


class ForecastMethod(enum.Enum):
    CONSTANT = "constant"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    EXPECTED = "expected"


class Event:
    def __init__(self, event_time: float, event_type: EventType, event_data):
        self.event_time = event_time
        self.event_type = event_type
        self.event_data = event_data

    def __lt__(self, other):
        return self.event_time < other.event_time


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
parser.add_argument("--sp_std", type=float, default=0.125)
parser.add_argument("--delay_std", type=float, default=0.125)
parser.add_argument("--assignment_policy", type=str,
                    choices=[x.value for x in AssignmentPolicy],
                    default=AssignmentPolicy.PREFERENCE.value,
                    help="Assignment method")
parser.add_argument("--forecast_method", type=str,
                    choices=[x.value for x in ForecastMethod],
                    default=ForecastMethod.PESSIMISTIC.value,
                    help="Method for forecasting waiting time, only relevant for preference assignment policy")
parser.add_argument("--pickup_distribution", nargs="+", type=float, default=[0.3, 0.25, 0.2, 0.15, 0.1],
                    help="Pickup distribution")
parser.add_argument("--arrival_rate", type=float, default=0.04, help="Arrival rate")
parser.add_argument("--days", type=int, default=50, help="Number of days to simulate")
parser.add_argument("--delivery_time", type=float, default=0.75, help="Delivery time")
parser.add_argument("--sp_data_file", type=str, default="data/linz-sp-map_pwl-0.5-0.04-300-10.csv", help="SP data file")
parser.add_argument("--dp_data_file", type=str, default="data/linz-dp-map_pwl-0.5-0.04-300-10.csv", help="DP data file")
args = parser.parse_args()

# Validate arguments
if round(sum(args.pickup_distribution), 5) != 1:
    raise ValueError("Pickup rates must sum to 1")
if args.delivery_time < 0 or args.delivery_time >= 1:
    raise ValueError("Delivery time must be between 0 and 1")
if args.days < 1:
    raise ValueError("Number of days must be positive")
if args.assignment_policy is not None and args.assignment_policy not in [x.value for x in AssignmentPolicy]:
    raise ValueError("Invalid assignment policy")
if (args.assignment_policy == AssignmentPolicy.PREFERENCE.value and
        args.forecast_method is not None and args.forecast_method not in [x.value for x in ForecastMethod]):
    raise ValueError("Invalid forecast method")
if args.assignment_policy == AssignmentPolicy.PREFERENCE.value and args.forecast_method is None:
    raise ValueError("Forecast method must be provided for preference-based assignment")

# Extract parameters
pickup_dist = args.pickup_distribution
arrival_rate = args.arrival_rate
days = args.days
delivery_time = args.delivery_time
assignment_policy: AssignmentPolicy = AssignmentPolicy(args.assignment_policy)
rule_based_assignment: bool = assignment_policy != AssignmentPolicy.PREFERENCE
preferences_based_assignment: bool = not rule_based_assignment
forecast_method: ForecastMethod = ForecastMethod(args.forecast_method) if preferences_based_assignment else None

# Load data
delivery_points, service_points = generate_data(args.dp_data_file, args.sp_data_file)

# Initialize simulation variables
seniority_pickup_rates = np.array(
    [pickup_dist[i] / np.sum(pickup_dist[i:])
     for i in range(len(pickup_dist))])
distances = cdist(delivery_points[['x', 'y']].values, service_points[['x', 'y']].values)  # distance matrix
n_service_points = len(service_points)  # number of service points
avg_pickup = round(np.sum([pickup_dist[i] * (i + 0.5) for i in range(len(pickup_dist))]), 5)
max_pickup = len(pickup_dist)  # number of days a parcel can remain at a service point
print(f"Average time until pickup: {avg_pickup}, max pickup: {max_pickup})")
assigned = [deque() for _ in range(n_service_points)]  # waiting parcels for each service point
sp_parcels = np.zeros((n_service_points, max_pickup), dtype=int)  # parcels at service points
parcel_statistics = np.empty((0, 16))
sp_statistics = np.empty((0, 3))  # day, sp, capacity used
availability_optimistic = np.zeros(n_service_points, dtype=int)
availability_pessimistic = np.zeros(n_service_points, dtype=int)
availability_expected = np.zeros(n_service_points, dtype=int)
dp_arrival_rates = delivery_points['Pop'] * arrival_rate
awaiting_assignments = 0  # number of parcels awaiting assignment
awaiting_delivery = 0  # number of parcels awaiting delivery
depot_parcels = []  # parcels at the depot


def optimistic_availability(sp):
    """ Calculate the minimum number of days until a service point has space for a new parcel """
    waiting_parcels = len(assigned[sp]) + 1  # +1 for the parcel being assigned
    return math.floor(waiting_parcels / service_points.iloc[sp]['capacity'])


def pessimistic_availability(sp):
    """ Calculate the maximum number of days until a service point has space for a new parcel """
    state = deque(sp_parcels[sp])
    waiting_parcels = len(assigned[sp]) + 1  # +1 for the parcel being assigned
    capacity = service_points.iloc[sp]['capacity']
    next_day_space = capacity - np.sum(state) + state.pop()
    max_days = 0  # days until space is available
    while waiting_parcels > next_day_space:
        max_days += 1
        waiting_parcels -= next_day_space
        state.appendleft(next_day_space)
        next_day_space = state.pop()
    return max_days


def expected_availability(sp, event_time):
    """ Estimate the number of days until a service point has space for a new parcel """
    capacity = service_points.iloc[sp]['capacity']
    state = sp_parcels[sp].copy()
    waiting_parcels = len(assigned[sp]) + 1
    expected_days = 0
    if max_pickup > 1:
        while True:
            # calculate the average number of parcels that will be picked up
            if expected_days == 0:
                # On the first day, adjust to the current time of day
                time_fraction = (next_delivery_time - event_time)
                state = state.astype('float64') * (1 - seniority_pickup_rates * time_fraction)
            else:
                state = state.astype('float64') * (1 - seniority_pickup_rates)
            space = int(round(capacity - np.sum(state)))
            state = state.round().astype(int)
            if waiting_parcels <= space:
                break
            state = np.roll(state, 1)
            state[0] = space
            waiting_parcels -= space
            expected_days += 1
    return expected_days


def update_availability_metrics(sp, event_time):
    """ Update the availability metrics for the service point """
    availability_optimistic[sp] = optimistic_availability(sp)
    availability_pessimistic[sp] = pessimistic_availability(sp)
    if expected_forecast:
        availability_expected[sp] = expected_availability(sp, event_time)


def disutility_f(d_ij, phi_ij=0):
    return ((1 + d_ij) * (1 + phi_ij)) ** 2


def disutility_g(t, gamma_it=0):
    return ((1 + t) * (1 + gamma_it)) ** 2


def disutility_function(d_ij, t, phi_ij=0, gamma_i=0):
    return disutility_f(d_ij, phi_ij) + disutility_g(t, gamma_i)


seed = 42
arrival_rng = np.random.default_rng(seed)
pickup_rng = np.random.default_rng(seed)
contact_rng = np.random.default_rng(seed)
next_parcel_id = 0

# Generate parcels for the entire simulation period, or load them from a file if they have been generated before
generated_parcels = []
pickle_file = f"data/parcels_{arrival_rate}.pkl"
if os.path.exists(pickle_file):
    print("Loading parcels from file...")
    with open(pickle_file, "rb") as f:
        generated_parcels = pickle.load(f)
else:
    print("Generating parcels...")
    for day in range(days):
        generated_parcels.append([])
        arrivals = arrival_rng.poisson(dp_arrival_rates) # number of parcels arriving at each delivery point
        for dp in range(len(delivery_points)):
            dp_arrivals = arrivals[dp]
            parcels_for_dp = []
            for _ in range(dp_arrivals):
                parcel = {
                    'id': next_parcel_id,
                    'arrival_time': day,
                    'delivery_point': dp,
                    'phi_base': arrival_rng.normal(0, 1, n_service_points),
                    'gamma_base': arrival_rng.normal(0, 1)
                }
                next_parcel_id += 1
                generated_parcels[day].append(parcel)
        # Shuffle the parcels to avoid bias
        arrival_rng.shuffle(generated_parcels[day])
    with open(pickle_file, "wb") as f:
        pickle.dump(generated_parcels, f)

# Adjust the disutility parameters for each parcel, gamma and phi should greater than -1
for parcels in generated_parcels:
    for parcel in parcels:
        epsilon = 0.0001
        parcel['gamma'] = max(args.delay_std * parcel['gamma_base'], -1 + epsilon)
        parcel['phi'] = np.maximum(args.sp_std * parcel['phi_base'], -1 + epsilon)


# Define output file names based on the simulation parameters, create output directories
method_str = assignment_policy.value + ("_" + str(forecast_method.value) if preferences_based_assignment else "")
city_name = parser.parse_args().dp_data_file.split("/")[-1].split("-")[0]
pickup_dist_str = f"{max_pickup}-{avg_pickup}"
output_dir = args.output_dir
log_dir = os.path.join(output_dir, "logs")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
base_file_name = f"{city_name}_{arrival_rate}_{pickup_dist_str}" \
                 f"_{args.sp_std}_{args.delay_std}" \
                 f"_{method_str}"

# If the assignment policy is 'lower_bound', calculate the optimal assignment for each parcel without simulation
calculate_lower_bound = assignment_policy == AssignmentPolicy.LOWER_BOUND
run_simulation = not calculate_lower_bound

if calculate_lower_bound:
    # Assign parcels to their optimal service point and delivery day (which minimizes their disutility),
    # without considering the capacity constraints of the service points
    for day in range(len(generated_parcels)):
        day_statistics = np.empty((0, 16))
        for parcel in generated_parcels[day]:
            dp = parcel['delivery_point']
            sp_disutilities = np.array([
                disutility_f(distances[dp][sp], parcel['phi'][sp])
                for sp in range(n_service_points)
            ])
            time_disutilities = np.array([
                disutility_g(time, parcel['gamma'])
                for time in range(20)
            ])
            sp = np.argmin(sp_disutilities)
            sp_distance = distances[dp][sp]
            time = np.argmin(time_disutilities)
            disutility = disutility_function(
                sp_distance,
                time,
                parcel['phi'][sp],
                parcel['gamma']
            )
            pack_statistics = [
                parcel['id'],
                day,
                -1,
                day + time,  # delivery day
                -1,  # forecast waiting time
                time,  # waiting time
                disutility,
                sp,
                service_points.iloc[sp]['capacity'],
                service_points.iloc[sp]['x'], service_points.iloc[sp]['y'],  # SP location
                dp,
                delivery_points.iloc[dp]['x'], delivery_points.iloc[dp]['y'],  # Recipient location
                delivery_points.iloc[dp]['Pop'],
                sp_distance
            ]
            day_statistics = np.append(day_statistics, np.array([pack_statistics]), axis=0)
        parcel_statistics = np.vstack((parcel_statistics, day_statistics))

# Initialize the event queue and schedule the first depot arrival and delivery events
event_queue: list[Event] = []
next_delivery_time = delivery_time
if run_simulation:
    initial_arrival_event = Event(0, EventType.DEPOT_ARRIVAL, None)
    heapq.heappush(event_queue, initial_arrival_event)
    initial_delivery_event = Event(next_delivery_time, EventType.DELIVERY, None)
    heapq.heappush(event_queue, initial_delivery_event)

# Setup for expected forecast update events
expected_forecast = preferences_based_assignment and forecast_method == ForecastMethod.EXPECTED
EXPECTED_FORECAST_UPDATE_INTERVAL = 0.01


# Run simulation - handle events until the queue is empty
last_day_pickups = 0
while len(event_queue) > 0:
    event = heapq.heappop(event_queue)
    event_day = int(event.event_time)
    if event.event_type == EventType.DEPOT_ARRIVAL:
        print(f"Processing warehouse arrival event at {event.event_time}")
        arrivals = generated_parcels[event_day]
        for parcel in arrivals:
            if preferences_based_assignment:
                # Schedule a customer contact event for the parcel, between now and the next delivery
                customer_response_time = contact_rng.uniform(0, next_delivery_time - event.event_time)
                parcel['customer_response_time'] = customer_response_time
                contact_event_time = event.event_time + customer_response_time
                contact_event = Event(contact_event_time, EventType.RECIPIENT_RESPONSE, parcel)
                heapq.heappush(event_queue, contact_event)
                awaiting_assignments += 1
            else:
                depot_parcels.append(parcel)

        if rule_based_assignment:
            print(f"Parcels arriving to the depot: {len(arrivals)}, total parcels: {len(depot_parcels)}")

        # Schedule the next arrival event
        if int(event.event_time) < days - 1:
            next_arrival = Event(event.event_time + 1, EventType.DEPOT_ARRIVAL, None)
            heapq.heappush(event_queue, next_arrival)

        if expected_forecast:
            check_time = event.event_time
            heapq.heappush(event_queue, Event(check_time, EventType.EXPECTED_FORECAST_UPDATE, None))

    elif event.event_type == EventType.RECIPIENT_RESPONSE:
        awaiting_assignments -= 1
        awaiting_delivery += 1
        parcel = event.event_data

        # Select service point
        if forecast_method == ForecastMethod.PESSIMISTIC:
            availability_forecast = availability_pessimistic
        elif forecast_method == ForecastMethod.OPTIMISTIC:
            availability_forecast = availability_optimistic
        elif forecast_method == ForecastMethod.CONSTANT:
            availability_forecast = np.zeros(n_service_points, dtype=int)
        else:
            availability_forecast = availability_expected

        # Estimate the disutility for each service point based on the forecast availability
        estimated_disutilities = [
            disutility_function(
                distances[parcel['delivery_point']][sp],
                availability_forecast[sp],
                parcel['phi'][sp],
                parcel['gamma']
            )
            for sp in range(n_service_points)
        ]
        # Choose the SP with the lowest estimated disutility
        sp = np.argmin(estimated_disutilities)

        parcel['service_point'] = sp
        parcel['forecast_waiting_time'] = availability_forecast[sp]
        assigned[sp].append(parcel)
        update_availability_metrics(sp, event.event_time)
    elif event.event_type == EventType.DELIVERY:
        print(f"Processing delivery event at {event.event_time}")
        if last_day_pickups > 0:
            print(f"{last_day_pickups} parcels have been picked up since the last delivery")
            last_day_pickups = 0
        print(f"Total parcels in SPs before delivery: {np.sum(sp_parcels)}, in depot: {len(depot_parcels)}")
        # Calculate daily metrics
        if rule_based_assignment:
            # Assign parcels to service points - when service point selection is not made by the customer
            if assignment_policy == AssignmentPolicy.BL1:
                # Baseline 1: Assign to the closest service point with available capacity for ensured next day delivery
                sp_free_space = np.array([
                    service_points.iloc[sp]['capacity'] - np.sum(sp_parcels[sp][:-1]) - len(assigned[sp])
                    for sp in range(n_service_points)], dtype=int)
                indices_with_capacity = np.nonzero(sp_free_space > 0)[0]
                while len(depot_parcels) > 0 and indices_with_capacity.size > 0:
                    parcel = depot_parcels.pop(0)
                    dp = parcel['delivery_point']
                    sp_index = np.argmin(distances[dp][indices_with_capacity])
                    sp = indices_with_capacity[sp_index]
                    sp_free_space[sp] -= 1
                    parcel['service_point'] = sp
                    assigned[sp].append(parcel)
                    indices_with_capacity = np.nonzero(sp_free_space)[0]
            elif assignment_policy == AssignmentPolicy.BL2:
                # In this policy, parcels are scanned in the same order as in BL1 but are
                # delivered only to the closest SP to their destination address. If this SP
                # has no available capacity, the parcel will wait in the depot for the next
                # delivery cycle.
                while len(depot_parcels) > 0:
                    parcel = depot_parcels.pop(0)
                    dp = parcel['delivery_point']
                    sp = np.argmin(distances[dp])
                    parcel['service_point'] = sp
                    assigned[sp].append(parcel)
            elif assignment_policy == AssignmentPolicy.BL3:
                # In this policy, just before each delivery cycle, the operator creates a fore-
                # cast of the newly available lockers of the SPs in the next few delivery
                # cycles and calculates the expected utility from assigning each parcel to
                # each SP in each cycle (based on each address and the number of cycles
                # passed since its arrival at the depot) using equation (1). Because g(t) is
                # convex, parcels that have already spent a long time in the depot are prior-
                # itized for earlier delivery cycles. The process is applied in a rolling horizon
                # fashion, so only the assignment plan of the first period is executed. The
                # assignment of parcels to future periods is tentative since it is based on an
                # uncertain forecast locker availability.

                n = len(depot_parcels)
                if n == 0:  # Skip
                    continue
                m = len(service_points)
                T = max_pickup
                horizon = T

                print("unassigned parcels", n, "horizon length", horizon)

                model = gp.Model()

                # Variables
                x = model.addVars(n, m, horizon, vtype=gp.GRB.BINARY, name="x")  # Assignment variables
                a = model.addVars(m, horizon, name="a")  # Parcels at SPs at day start

                # Constraints
                capacities = service_points['capacity'].values
                initial_parcels = np.sum(sp_parcels, axis=1)
                daily_pickups = np.flip(sp_parcels, axis=1)
                model.addConstrs((a[j, t] == (initial_parcels[j] if t == 0 else a[j, t - 1])
                                  - daily_pickups[j, t] + x.sum("*", j, t)
                                  for t in range(horizon) for j in range(m)), name="parcels_at_sp")
                model.addConstrs((a[j, t] <= capacities[j] for t in range(horizon) for j in range(m)), name="capacity")
                model.addConstrs((x.sum(i, '*', '*') == 1 for i in range(n)), name="one_assignment")

                # Objective function
                pack_dps = [parcel['delivery_point'] for parcel in depot_parcels]
                pack_arrivals = [parcel['arrival_time'] for parcel in depot_parcels]
                pack_distances = np.array(
                    [[disutility_f(distances[pack_dps[i]][j]) for j in range(m)] for i in range(n)])
                pack_delays = np.array(
                    [[disutility_g(t + event_day - pack_arrivals[i]) for i in range(n)] for t in range(horizon)])
                objective = gp.quicksum(
                    x[i, j, t] * (
                            pack_distances[i][j] +  # Distance cost
                            pack_delays[t][i]  # Delay cost
                    )
                    for i in range(n) for j in range(m) for t in range(horizon)
                )
                model.setObjective(objective, gp.GRB.MINIMIZE)

                # Solve model
                model.setParam('OutputFlag', 1)
                model.setParam('LogFile', f"{log_dir}/{base_file_name}_{event_day}.log")
                model.optimize()

                # Extract results and assign parcels
                x_sol = model.getAttr('x', x)
                assigned_parcels = []

                for name, val in zip(x_sol.keys(), x_sol.values()):
                    i, j, t = name
                    if t == 0 and val == 1:  # Only consider the first period
                        parcel = depot_parcels[i]
                        parcel['service_point'] = j
                        assigned_parcels.append(i)
                        assigned[j].append(parcel)

                # Remove assigned parcels from the depot
                depot_parcels = [parcel for i, parcel in enumerate(depot_parcels) if i not in assigned_parcels]
                print(f"Assigned {len(assigned_parcels)} parcels, {len(depot_parcels)} delayed")

        for sp in range(n_service_points):
            update_availability_metrics(sp, event.event_time)
        delivery_statistics = np.empty((0, 16))
        for sp in range(n_service_points):
            sp_parcels[sp, -1] = 0  # Dump parcels that were not picked up
            sp_info = service_points.iloc[sp]
            sp_capacity = int(sp_info['capacity'])
            sp_available_capacity = sp_capacity - np.sum(sp_parcels[sp])
            sp_assigned_parcels = len(assigned[sp])
            delivered = max(int(min(sp_available_capacity, sp_assigned_parcels)), 0)

            # print(f"SP {sp} has {sp_available_capacity} available capacity and {len(assigned[sp])} assigned parcels")
            if delivered > 0:
                delivery_time = event.event_time
                awaiting_delivery -= delivered
                for _ in range(delivered):
                    parcel = assigned[sp].popleft()
                    parcel['delivery_time'] = delivery_time
                    days_to_pickup = pickup_rng.choice(max_pickup, p=pickup_dist)
                    parcel['seniority_at_pickup'] = days_to_pickup
                    pickup_time = delivery_time + days_to_pickup + pickup_rng.random()  # Delivery time + random days to pickup + random time during the day
                    pickup_event = Event(pickup_time, EventType.PARCEL_PICKUP, parcel)
                    heapq.heappush(event_queue, pickup_event)
                    # Add parcel data
                    arrival_day = parcel['arrival_time']
                    delay = event_day - arrival_day
                    parcel['waiting_time'] = delay

                    # Calculate disutility
                    sp = parcel['service_point']
                    distance = distances[parcel['delivery_point']][sp]
                    disutility = disutility_function(distance,
                                                     delay,
                                                     parcel['phi'][sp],
                                                     parcel['gamma'],
                                                     )
                    parcel['disutility'] = disutility
                    if preferences_based_assignment:
                        forecast_waiting_time = parcel['forecast_waiting_time']
                    else:
                        forecast_waiting_time = -1
                    # Add statistics
                    parcel_dp = parcel['delivery_point']
                    dp_info = delivery_points.iloc[parcel_dp]
                    customer_response_time = parcel['customer_response_time'] if preferences_based_assignment else -1
                    pack_statistics = [
                        parcel['id'],
                        arrival_day,
                        customer_response_time,
                        event_day,
                        forecast_waiting_time,
                        delay,
                        disutility,
                        sp,
                        sp_info['capacity'],
                        sp_info['x'], sp_info['y'],  # SP location
                        parcel_dp,
                        dp_info['x'], dp_info['y'],  # Recipient location
                        dp_info['Pop'],
                        distance
                    ]
                    delivery_statistics = np.append(delivery_statistics, np.array([pack_statistics]), axis=0)
            # Update SP state
            sp_parcels[sp, 1:] = sp_parcels[sp, :-1]
            sp_parcels[sp, 0] = delivered
            update_availability_metrics(sp, event.event_time)
            sp_usage = np.sum(sp_parcels[sp]) / sp_capacity
            sp_stats = [event_day, sp, sp_usage]
            sp_statistics = np.append(sp_statistics, np.array([sp_stats]), axis=0)
        parcel_statistics = np.vstack((parcel_statistics, delivery_statistics))

        total_assigned_parcels = np.sum([len(assigned[sp]) for sp in range(n_service_points)])
        if event_day < days - 1 or awaiting_assignments > 0 or awaiting_delivery > 0 or total_assigned_parcels > 0\
                or len(depot_parcels) > 0:
            next_delivery_time += 1
            next_delivery = Event(next_delivery_time, EventType.DELIVERY, None)
            heapq.heappush(event_queue, next_delivery)
        print(f"Total parcels in SPs after delivery: {np.sum(sp_parcels)}, in depot: {len(depot_parcels)}")
    elif event.event_type == EventType.PARCEL_PICKUP:
        last_day_pickups += 1
        parcel = event.event_data
        sp = parcel['service_point']
        seniority = parcel['seniority_at_pickup']
        sp_parcels[sp, seniority] -= 1
        update_availability_metrics(sp, event.event_time)
    elif event.event_type == EventType.EXPECTED_FORECAST_UPDATE:
        # If there are parcels awaiting assignment, check availability and schedule next check
        if awaiting_assignments > 0:
            for sp in range(n_service_points):
                update_availability_metrics(sp, event.event_time)
            next_check = Event(event.event_time + EXPECTED_FORECAST_UPDATE_INTERVAL, EventType.EXPECTED_FORECAST_UPDATE, None)
            heapq.heappush(event_queue, next_check)

# Save parcel statistics
parcels_stats_file = f"{output_dir}/{base_file_name}.csv"
parcels_stats_header = ['id', 'arrival_day', 'customer_response_time', 'delivery_day', 'forecast_waiting_time', 'waiting_time',
                        'disutility',
                        'sp', 'sp_capacity', 'sp_loc_x', 'sp_loc_y', 'dp', 'dp_loc_x', 'dp_loc_y', 'dp_population',
                        'distance']
parcels_stats_fmt = "%d,%d,%f,%d,%d,%d,%f,%d,%d,%f,%f,%d,%f,%f,%d,%f"
parcel_statistics = parcel_statistics[parcel_statistics[:, 0].argsort()]
parcels_stats_flat = parcel_statistics.reshape(-1, parcel_statistics.shape[-1])
with open(parcels_stats_file, 'w') as f:
    f.write(",".join(parcels_stats_header) + "\n")
    np.savetxt(f, parcels_stats_flat, fmt=parcels_stats_fmt, delimiter=",")

# Save SP statistics
sp_stats_file = f"{output_dir}/{base_file_name}_sp.csv"
sp_stats_header = ['day', 'sp', 'usage']
sp_stats_fmt = "%d,%d,%f"
sp_statistics = sp_statistics[sp_statistics[:, 0].argsort()]
sp_stats_flat = sp_statistics.reshape(-1, sp_statistics.shape[-1])
with open(sp_stats_file, 'w') as f:
    f.write(",".join(sp_stats_header) + "\n")
    np.savetxt(f, sp_stats_flat, fmt=sp_stats_fmt, delimiter=",")
