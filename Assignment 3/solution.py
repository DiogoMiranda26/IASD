import search
import numpy as np
import copy
from collections import defaultdict
from itertools import permutations

class FleetProblem(search.Problem):
    def __init__(self):
        self.A_matrix = None # Transportation time matrix
        self.R_list = [] # List of the requests
        self.V_list = [] # List of the number of seats for each vehicle
        
    def get_request_time(self, request):
        """Return the time of the request"""
        return self.R_list[request][0]
    
    def get_origin(self, request):
        """Return the origin of the request"""
        return self.R_list[request][1]
    
    def get_destination(self, request):
        """Return the destination of the request"""
        return self.R_list[request][2]
    
    def get_passengers(self, request):
        """Return the number of passengers of the request"""
        return self.R_list[request][3]
    
    def get_transportation_time(self, origin, destination):
        """Return the transportation time from origin to destination point"""
        return self.A_matrix[origin][destination]
    
    # For a given state, this class contains the information for all vehicles
    class VehiclesData():
        def __init__(self, V_list, R_list):
            # List that stores the current position of each vehicle 
            self.vehicles_position = [0] * len(V_list)
            # List that stores the available seats for each vehicle at a given time
            self.available_seats = copy.deepcopy(V_list)
            # List that stores the internal time of each vehicle
            self.vehicles_clock = [0] * len(V_list)
            # List that stores the vehicle that answers request index
            self.request_vehicle_list = [-1] * len(R_list)
            # List that stores the pickup time of the requests
            self.pickup_times = [0] * len(R_list)
            # List that stores the pickup delays of the requests
            self.pickup_delays = [0] * len(R_list)
            # List that stores the dropoff delays of the requests
            self.dropoff_delays = [0] * len(R_list)
            # List that stores the time of arrival of a vehicle at any position
            self.arrivals = [0] * len(V_list)
     
    # Class that represents the state structure of the problem       
    class State():
        def __init__(self, R_list, vehicles_data):
            self.info = vehicles_data
            # List that stores the status of each request
            # The status of a request changes from 'Waiting' to 'Onboard' and from 'Onboard' to 'Finished'
            self.requests = ['Waiting'] * len(R_list)
        
    def classToTuple(self, state):
        """Convert class State into a tuple"""
        tuple_state = (
            tuple(state.info.vehicles_position), 
            tuple(state.info.available_seats),
            tuple(state.info.vehicles_clock),
            tuple(state.info.request_vehicle_list),
            tuple(state.requests),
            tuple(state.info.pickup_times),
            tuple(state.info.pickup_delays),
            tuple(state.info.dropoff_delays),
            tuple(state.info.arrivals)
            )
        return tuple_state
    
    def tupleToClass(self, tuple_state):
        """Convert tuple into a class State"""
        myState = self.State(self.R_list, self.VehiclesData(self.V_list, self.R_list))
        myState.info.vehicles_position = list(tuple_state[0])
        myState.info.available_seats = list(tuple_state[1])
        myState.info.vehicles_clock = list(tuple_state[2])
        myState.info.request_vehicle_list = list(tuple_state[3])
        myState.requests = list(tuple_state[4])
        myState.info.pickup_times = list(tuple_state[5])
        myState.info.pickup_delays = list(tuple_state[6])
        myState.info.dropoff_delays = list(tuple_state[7])
        myState.info.arrivals = list(tuple_state[8])
        return myState
        
    def load(self, fh):
        """Load a problem from the opened file object fh"""
        file = fh
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                continue
            elif line.startswith('P'):
                _, number_of_points = line.split()
                number_of_points = int(number_of_points)
                self.A_matrix = np.zeros((number_of_points, number_of_points), dtype = float)
                for i in range(number_of_points - 1):
                    # Read the transportation times and add it adequately
                    self.A_matrix[i, i+1:] = list(map(float, file.readline().strip().split()))
                # Turn the matrix into a symmetric matrix
                self.A_matrix = self.A_matrix + self.A_matrix.T 
            elif line.startswith('R'):
                _, number_of_requests = line.split()
                number_of_requests = int(number_of_requests)
                for _ in range(number_of_requests):
                    request = tuple(map(float, file.readline().strip().split()))
                    request = (request[0], int(request[1]), int(request[2]), int(request[3]))
                    self.R_list.append(request)
            elif line.startswith('V'):
                _, number_of_vehicles = line.split()
                number_of_vehicles = int(number_of_vehicles)
                for _ in range(number_of_vehicles):
                    number_of_seats = int(file.readline().strip())
                    self.V_list.append(number_of_seats)
        # Initial state where all passengers are waiting for a pickup
        vehicles_data = self.VehiclesData(self.V_list, self.R_list)  
        initial_state = self.State(self.R_list, vehicles_data)
        initial_state = self.classToTuple(initial_state)
        # Goal state where all requests have been attended
        goal_state = tuple(['Finished'] * len(self.R_list))
        super().__init__(initial_state, goal_state)
        return
    
    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from state1 via action,
        assuming cost c to get up to state1"""
        state = self.tupleToClass(state1)
        operator, _ , request, predicted_time = action
        pick_up_time = state.info.pickup_times[request]
        delay = 0
        if operator == 'Dropoff':
            delay = predicted_time - pick_up_time - self.get_transportation_time(self.get_origin(request), self.get_destination(request))
        elif operator == 'Pickup':
            delay = predicted_time - self.get_request_time(request)
        cost = c + delay
        return cost
    
    def stepsLeft(self, state):
        """Return the number of actions left for the current state to reach the solution"""
        steps = 0
        for status in state.requests:
            if status == 'Waiting':
                steps +=2
            elif status == 'Onboard':
                steps += 1
        return steps
        
    def h(self, state):
        """Return the heuristic value for the given state"""
        node = state
        state = node.state
        state = self.tupleToClass(state)
        estimated_cost = 0
        waiting_requests = [request for request, status in enumerate(state.requests) if status == 'Waiting']
        onboard_requests = [request for request, status in enumerate(state.requests) if status == 'Onboard']
        finished_requests = [request for request, status in enumerate(state.requests) if status == 'Finished']
        estimated_cost_onboard = 0
        estimated_cost_waiting = 0
        cost_finished = 0
        vehicle_requests = defaultdict(list) # Dictionary that stores the requests onboard for each vehicle
        for request, vehicle in enumerate(state.info.request_vehicle_list):
            if vehicle != -1:
                vehicle_requests[vehicle].append(request)
        for request in finished_requests:
            cost_finished += state.info.dropoff_delays[request]
        for request in onboard_requests:
            vehicle = state.info.request_vehicle_list[request]
            dropoff_time = state.info.vehicles_clock[vehicle] + self.get_transportation_time(state.info.vehicles_position[vehicle], self.get_destination(request))
            expected_dropoff_time = state.info.pickup_times[request] + self.get_transportation_time(self.get_origin(request), self.get_destination(request))
            delay = dropoff_time - expected_dropoff_time + state.info.pickup_delays[request]
            estimated_cost_onboard += delay
        if waiting_requests:
            for request in waiting_requests:
                delay_vehicles = [float('inf')] * len(self.V_list)
                for vehicle, seats in enumerate(self.V_list):
                    # The vehicle can pick up the request
                    if state.info.available_seats[vehicle] >= self.get_passengers(request):
                        pickup_time = state.info.vehicles_clock[vehicle] + self.get_transportation_time(state.info.vehicles_position[vehicle], self.get_origin(request))
                        delay = max(pickup_time - self.get_request_time(request), 0)
                        # select the vehicle with minimum pickup delay to serve the request
                        delay_vehicles[vehicle] = delay
                    # The vehicle cannot pick up the request until it drops some passengers
                    elif state.info.available_seats[vehicle] < self.get_passengers(request) and seats >= self.get_passengers(request):
                        permutations_list = list(permutations(vehicle_requests[vehicle]))
                        delay_permutations = [float('inf')] * len(permutations_list)
                        for permutation in permutations_list:
                            vehicle_clock = float(state.info.vehicles_clock[vehicle])
                            vehicle_position = int(state.info.vehicles_position[vehicle])
                            available_seats = int(state.info.available_seats[vehicle])
                            for index, dropped in enumerate(permutation):
                                vehicle_clock = vehicle_clock + self.get_transportation_time(vehicle_position, self.get_destination(dropped))
                                vehicle_position = self.get_destination(dropped)
                                available_seats += self.get_passengers(dropped)
                                if available_seats >= self.get_passengers(request):
                                    pickup_time = vehicle_clock + self.get_transportation_time(vehicle_position, self.get_origin(request))
                                    delay = max(pickup_time - self.get_request_time(request), 0)
                                    delay_permutations[index] = delay
                                    break
                        delay_vehicles[vehicle] = min(delay_permutations)
                        if delay_vehicles[vehicle] == 0:
                            break
                estimated_cost_waiting += min(delay_vehicles)
        if len(onboard_requests) == 1 and len(waiting_requests) == 0:
            estimated_cost = estimated_cost_onboard + estimated_cost_waiting + cost_finished
            pass   
        estimated_cost = estimated_cost_onboard + estimated_cost_waiting + cost_finished
        estimated_cost += (2*len(self.R_list) - self.stepsLeft(state)) * np.amin(self.A_matrix[np.triu_indices(len(self.A_matrix), k = 1)])/1000
        return estimated_cost
    
    def result(self, state, action):
        """Return the state that results from executing the given action in the given state""" 
        operator, vehicle, request, action_time = action
        state = self.tupleToClass(state)
        new_state = copy.deepcopy(state)
        if operator == 'Pickup':
            new_state.requests[request] = 'Onboard'
            # The vehicle drives to the request position
            if self.get_origin(request) != state.info.vehicles_position[vehicle]:
                new_state.info.arrivals[vehicle] = state.info.vehicles_clock[vehicle] + self.get_transportation_time(state.info.vehicles_position[vehicle], self.get_origin(request))
                new_state.info.vehicles_position[vehicle] = self.get_origin(request)
            new_state.info.request_vehicle_list[request] = vehicle
            # If the vehicle serves a request at the current position, then it only updates its internal clock if it has to wait for the request 
            # If the vehicle serves a request at a different position, then it naturally updates its internal clock
            new_state.info.vehicles_clock[vehicle] = max(new_state.info.vehicles_clock[vehicle], action_time)
            new_state.info.available_seats[vehicle] -= self.get_passengers(request)
            new_state.info.pickup_times[request] = action_time
            new_state.info.pickup_delays[request] = max(action_time - self.get_request_time(request), 0)
        elif operator == 'Dropoff':
            new_state.requests[request] = 'Finished'
            if self.get_destination(request) != state.info.vehicles_position[vehicle]:
                new_state.info.arrivals[vehicle] = action_time
                new_state.info.vehicles_position[vehicle] = self.get_destination(request)
            new_state.info.vehicles_clock[vehicle] = action_time
            new_state.info.available_seats[vehicle] += self.get_passengers(request)
            new_state.info.request_vehicle_list[request] = -1
            expected_dropoff = self.get_request_time(request) + self.get_transportation_time(self.get_origin(request), self.get_destination(request))
            new_state.info.dropoff_delays[request] = action_time - expected_dropoff
        # print(f'The new state will be {new_state.requests} after the action {action}')
        new_state = self.classToTuple(new_state)
        return new_state

    def actions(self, state):
        """Return the list of actions that can be executed in the given state"""
        actions = []
        state = self.tupleToClass(state)
        for request, request_state in enumerate(state.requests):
            if request_state == 'Waiting':
                for vehicle in range(len(self.V_list)):
                    # Pickup action: valid if the vehicle has enough seats
                    if state.info.available_seats[vehicle] >= self.get_passengers(request):
                        transportation_time = self.get_transportation_time(state.info.vehicles_position[vehicle], self.get_origin(request))
                        # The vehicle serves more than one request at a position
                        if transportation_time == 0:
                            if self.get_request_time(request) <= state.info.arrivals[vehicle]:
                                action_time = state.info.arrivals[vehicle]
                            else:
                                action_time = self.get_request_time(request)
                        else: 
                            action_time = max(self.get_request_time(request), state.info.vehicles_clock[vehicle] + transportation_time)
                        actions.append(('Pickup', vehicle, request, action_time))
            # Dropoff action
            elif request_state == 'Onboard':
                vehicle = state.info.request_vehicle_list[request]
                transportation_time = self.get_transportation_time(state.info.vehicles_position[vehicle], self.get_destination(request))
                action_time = state.info.vehicles_clock[vehicle] + transportation_time
                actions.append(('Dropoff', vehicle, request, action_time))
        state = self.classToTuple(state)
        return actions
        
    def goal_test(self, state):
        """Return True if the state is a goal"""
        return True if state[4] == self.goal else False
         
    def solve(self):
        """Calls the uniform_cost_search method from search.py and returns the solution"""
        node = search.Node(self.initial)
        node = search.astar_search(self)
        return node.solution()