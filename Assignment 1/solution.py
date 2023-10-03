import search
import numpy as np
from collections import defaultdict

class FleetProblem(search.Problem):
    def __init__(self):
        self.A_matrix = None
        self.R_list = list()
        self.V_list = list()

    def load(self, fh):
        # Load a problem from the opened file object fh
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
        return
    
    # return the time of the request
    def get_request_time(self, request):
        return self.R_list[request - 1][0]
    
    # return the transportation time of the request from origin point to destination point
    def get_transportation_time(self, request):
        pickup_point_index = self.R_list[request - 1][1]
        dropoff_point_index = self.R_list[request - 1][2]
        return self.A_matrix[pickup_point_index][dropoff_point_index]
    
    # return the sum of all delays
    def cost(self, sol):
        vehicle_requests = defaultdict(list)
        dropoff_times = defaultdict(float)
        # loop to read the solutions
        for s in sol:
            action, vehicle, request, time_of_action = s[0], s[1], s[2], s[3]
            if request not in vehicle_requests[vehicle]:
                vehicle_requests[vehicle].append(request)
            if action == 'Dropoff':
                dropoff_times[request] = time_of_action
        J = 0 # sum of all delays
        # loop to calculate the cost
        for vehicle, requests in vehicle_requests.items():
            for request in requests:
                dropoff_time = dropoff_times[request]
                request_time = self.get_request_time(request)
                transportation_time = self.get_transportation_time(request)
                delay = dropoff_time - request_time - transportation_time
                J = J + delay     
        return J
         