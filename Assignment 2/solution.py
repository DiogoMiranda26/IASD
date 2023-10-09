import search
import numpy as np
import copy
from collections import defaultdict

class FleetProblem(search.Problem):
    def __init__(self):
        self.A_matrix = None # Transportation time matrix
        self.R_list = list() # List of the requests
        self.V_list = list() # List of the number of seats for each vehicle
    
    # For a given state, this class returns the information for each vehicle
    class VehiclesData():
        def __init__(self, V_list, R_list):
            self.vehicles_position = [0] * len(V_list) # List that stores the vehicle that carries request index   
            self.available_seats = copy.deepcopy(V_list) # List that stores the available seats for each vehicle at a given time
            self.vehicles_clock = [0] * len(V_list) # List that stores the current position of each vehicle
            self.request_vehicle_list = [-1] * len(R_list) # List that stores the time instance at the current position of each vehicle
            
    class State():
        def __init__(self, V_list, R_list, vehicles_data):
            self.info = vehicles_data
            self.requests = ['Waiting'] * len(R_list)
            self.time = max(vehicles_data.vehicles_clock)
            
        def __lt__(self, other_state):
            return self.time < other_state.time
      
    # Load a problem from the opened file object fh
    def load(self, fh):
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
                    # read the transportation times and add it adequately
                    self.A_matrix[i, i+1:] = list(map(float, file.readline().strip().split()))
                # turn the matrix into a symmetric matrix
                self.A_matrix = self.A_matrix + self.A_matrix.T 
            elif line.startswith('R'):
                _, number_of_requests = line.split()
                number_of_requests = int(number_of_requests)
                for _ in range(number_of_requests):
                    # read the request
                    request = tuple(map(float, file.readline().strip().split()))
                    request = (request[0], int(request[1]), int(request[2]), int(request[3]))
                    self.R_list.append(request)
            elif line.startswith('V'):
                _, number_of_vehicles = line.split()
                number_of_vehicles = int(number_of_vehicles)
                for _ in range(number_of_vehicles):
                    # read the number of seats for a vehicle
                    number_of_seats = int(file.readline().strip())
                    self.V_list.append(number_of_seats)
 
        # Initial state where all passengers are waiting for a pickup
        vehicles_data = self.VehiclesData(self.V_list, self.R_list)  
        initial_state = self.State(self.V_list, self.R_list, vehicles_data)
        # Goal state where all requests have been attended  
        goal_state = ['Finished'] * len(self.R_list)
        super().__init__(initial_state, goal_state)
        return
    
    # return the time of the request
    def get_request_time(self, request):
        return self.R_list[request - 1][0]
    
    # return the transportation time of the request from origin point to destination point
    def get_transportation_time(self, origin, destination):
        return self.A_matrix[origin][destination]
   
    # Return the state that results from executing the given action in the given state.
    # The new state must change from 'Waiting' to 'Onboard' and from 'Onboard' to 'Finished'     
    def result(self, state, action):
        operator, vehicle, request, action_time = action
        # print(f'This is the state {state.requests}')
        new_state = copy.deepcopy(state)
        if operator == 'Pickup':
            new_state.requests[request] = 'Onboard'
            new_state.info.request_vehicle_list[request] = vehicle
            new_state.info.vehicles_clock[vehicle] = action_time
            new_state.info.vehicles_position[vehicle] = self.R_list[request][1]
            new_state.info.available_seats[vehicle] -= self.R_list[request][3]
            new_state.time = action_time
        elif operator == 'Dropoff':
            new_state.requests[request] = 'Finished'
            new_state.info.vehicles_clock[vehicle] = action_time
            new_state.info.vehicles_position[vehicle] = self.R_list[request][2]
            new_state.info.available_seats[vehicle] += self.R_list[request][3]
            new_state.time = action_time
        # print(f'The new state will be {new_state.requests} after the action {action}')
        return new_state

    # Return the list of actions that can be executed in the given state
    def actions(self, state):
        actions = []
        for request, request_state in enumerate(state.requests):
            if request_state == 'Waiting':
                for vehicle in range(len(self.V_list)):
                    # Pickup action: valid if the vehicle has enough seats
                    if state.info.available_seats[vehicle] >= self.R_list[request][3]:
                        # The action time is the clock up to date plus the transportation time from the current vehicle position to the request origin point
                        action_time = max(self.R_list[request][0], state.info.vehicles_clock[vehicle] + self.get_transportation_time(state.info.vehicles_position[vehicle], self.R_list[request][1]))
                        actions.append(('Pickup', vehicle, request, action_time))
            # Dropoff action
            elif request_state == 'Onboard':
                vehicle = state.info.request_vehicle_list[request]
                # The action time is the clock up to date plus the transportation time from the current vehicle position to the request destination point
                action_time = state.info.vehicles_clock[vehicle] + self.get_transportation_time(state.info.vehicles_position[vehicle], self.R_list[request][2])
                actions.append(('Dropoff', vehicle, request, action_time))
        # print(f'Possible actions : {actions}')
        # print()
        # if not hasattr(self, 'actions_counter'):
        #     self.actions_counter = 0
        # self.actions_counter += 1
        # if self.actions_counter == 2:
        #     exit()
        return actions
        
    def goal_test(self, state):
        return True if state.requests == self.goal else False
    
    def path_cost(self, c, state1, action, state2):
        return state2.time
        
    
    def solve(self):
        # Calls the uninformed search algorithm chosen
        # Returns a solution using the specified format
        node = search.Node(self.initial)
        node = search.uniform_cost_search(self)
        return node.solution()
           
         