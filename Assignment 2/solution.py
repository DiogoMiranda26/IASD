import search
import numpy as np
import copy
from collections import defaultdict

class FleetProblem(search.Problem):
    def __init__(self):
        self.A_matrix = None # Transportation time matrix
        self.R_list = [] # List of the requests
        self.V_list = [] # List of the number of seats for each vehicle
    
    # For a given state, this class contains the information for all vehicles
    class VehiclesData():
        def __init__(self, V_list, R_list):
            # List that stores the vehicle that carries request index 
            self.vehicles_position = [0] * len(V_list)
            # List that stores the available seats for each vehicle at a given time
            self.available_seats = copy.deepcopy(V_list)
            # List that stores the current position of each vehicle
            self.vehicles_clock = [0] * len(V_list)
            # List that stores the vehicle that answers request index
            self.request_vehicle_list = [-1] * len(R_list)
            # List that stores the pickup time of the requests
            self.timing_list = [0] * len(R_list)
     
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
            tuple(state.info.timing_list)
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
        myState.info.timing_list = list(tuple_state[5])
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
    
    def get_request_time(self, request):
        """Return the time of the request"""
        return self.R_list[request - 1][0]
    
    def get_transportation_time(self, origin, destination):
        """Return the transportation time from origin to destination point"""
        return self.A_matrix[origin][destination]
    
    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from state1 via action,
        assuming cost c to get up to state1"""
        state = self.tupleToClass(state1)
        operator, _ , request, predicted_time = action
        pick_up_time = state.info.timing_list[request]
        delay = 0
        if operator == 'Dropoff':
            delay = predicted_time - pick_up_time - self.get_transportation_time(self.R_list[request][1], self.R_list[request][2])
        elif operator == 'Pickup':
            delay = predicted_time - self.get_request_time(request)
        cost = c + delay
        return cost
        
    def result(self, state, action):
        """Return the state that results from executing the given action in the given state""" 
        operator, vehicle, request, action_time = action
        state = self.tupleToClass(state)
        new_state = copy.deepcopy(state)
        if operator == 'Pickup':
            new_state.requests[request] = 'Onboard'
            new_state.info.request_vehicle_list[request] = vehicle
            new_state.info.vehicles_clock[vehicle] = action_time
            new_state.info.vehicles_position[vehicle] = self.R_list[request][1]
            new_state.info.available_seats[vehicle] -= self.R_list[request][3]
            new_state.info.timing_list[request] = action_time
        elif operator == 'Dropoff':
            new_state.requests[request] = 'Finished'
            new_state.info.vehicles_clock[vehicle] = action_time
            new_state.info.vehicles_position[vehicle] = self.R_list[request][2]
            new_state.info.available_seats[vehicle] += self.R_list[request][3]
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
                    if state.info.available_seats[vehicle] >= self.R_list[request][3]:
                        transportation_time = self.get_transportation_time(state.info.vehicles_position[vehicle], self.R_list[request][1])
                        action_time = max(self.R_list[request][0], state.info.vehicles_clock[vehicle] + transportation_time)
                        actions.append(('Pickup', vehicle, request, action_time))
            # Dropoff action
            elif request_state == 'Onboard':
                vehicle = state.info.request_vehicle_list[request]
                transportation_time = self.get_transportation_time(state.info.vehicles_position[vehicle], self.R_list[request][2])
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
        node = search.uniform_cost_search(self)
        return node.solution()
           
         