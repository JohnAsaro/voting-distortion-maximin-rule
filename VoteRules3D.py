from numpy import random
import numpy as np
import scipy as sp
import math
from vote import Vote
from election import Election
import matplotlib.pyplot as plt 

class SVoter3D:
    def __init__(self, num, x, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
        
        self.id = num
        self.scores = {}

    def setScores(self,scoreDict):
        self.scores = scoreDict

    def __str__(self):
        return "Voter "+str(self.id)

class SCandidate3D:
    def __init__(self, num, x, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.id = num

    def __str__(self):
        return "Candidate "+str(self.id)
    
class VoteResult3D:
    def __init__(self, n, m, dimension = "1D", distribution="normal"):
        self.voters = []      #size of voters is n
        self.candidates = []  #size of candidates is m
        self.distribution = distribution
        self.dimension = dimension

        #generate random coordinates of voters and candidates for different distributions
        if self.distribution == "normal":
            x_voters = random.normal(50, 18,n)
            x_candidates = random.normal(50, 18, m)
            y_voters = random.normal(50, 18,n)
            y_candidates = random.normal(50, 18, m)
            z_voters = random.normal(50, 18,n)
            z_candidates = random.normal(50, 18, m)
            
            
        elif self.distribution == "poisson":
            x_voters = random.poisson(30, n)
            x_candidates = random.poisson(30, m)
            y_voters = random.poisson(30, n)
            y_candidates = random.poisson(30, m)
            z_voters = random.poisson(30, n)
            z_candidates = random.poisson(30, m)
           

        elif self.distribution == "uniform":
            x_voters = random.uniform(0, 100, n)
            x_candidates = random.uniform(0, 100, m)
            y_voters = random.uniform(0, 100, n)
            y_candidates = random.uniform(0, 100, m)
            z_voters = random.uniform(0, 100, n)
            z_candidates = random.uniform(0, 100, m)
            

        elif self.distribution == "bimodal":
            x_voters1 = random.normal(30, 10, n//2)
            x_voters2 = random.normal(70, 10, n-n//2)
            y_voters1 = random.normal(30,10,n//2)
            y_voters2 = random.normal(70, 10, n-n//2)
            z_voters1 = random.normal(30,10,n//2)
            z_voters2 = random.normal(70, 10, n-n//2)

            x_candidates1 = random.normal(30, 10, m//2)   
            x_candidates2 = random.normal(70, 10, m-m//2)
            y_candidates1 = random.normal(30, 10, m//2)
            y_candidates2 = random.normal(70, 10, m-m//2)
            z_candidates1 = random.normal(30, 10, m//2)
            z_candidates2 = random.normal(70, 10, m-m//2)

            x_voters = np.concatenate((x_voters1, x_voters2), axis=None)
            y_voters = np.concatenate((y_voters1, y_voters2), axis=None)
            z_voters = np.concatenate((z_voters1, z_voters2), axis=None)

            x_candidates = np.concatenate((x_candidates1, x_candidates2), axis = None)
            y_candidates = np.concatenate((y_candidates1, y_candidates2), axis = None)
            z_candidates = np.concatenate((z_candidates1, z_candidates2), axis = None)

        #generate voters and candidates based on the coordinates
        for i in range(n):
            voter = None
            if self.dimension == "1D":
                voter = SVoter3D(i, x_voters[i])
            elif self.dimension == "2D":
                voter = SVoter3D(i, x_voters[i], y_voters[i])
            elif self.dimension == "3D":
                voter = SVoter3D(i, x_voters[i], y_voters[i], z_voters[i])
            
            self.voters.append(voter)

        for i in range(m):
            candidate = None
            if self.dimension == "1D":
                candidate = SCandidate3D(i, x_candidates[i])
            elif self.dimension == "2D":
                candidate = SCandidate3D(i, x_candidates[i], y_candidates[i])
            elif self.dimension == "3D":
                candidate = SCandidate3D(i, x_candidates[i], y_candidates[i], z_candidates[i])
            
            self.candidates.append(candidate)


        self.minDistance = float('inf')
        self.OPTcandidate = self.candidates[0]
        for candidate in self.candidates:
            sumDistance = 0
            for voter in self.voters:
                distance = math.sqrt((voter.x - candidate.x) ** 2 + (voter.y - candidate.y) ** 2 + (voter.z - candidate.z) ** 2)
                sumDistance += distance
            if sumDistance < self.minDistance:
                self.minDistance = sumDistance
                self.OPTcandidate = candidate
       
        #get preference profile of each voter given a set of candidates
        self.ballots = []
        for voter in self.voters:
            distances = {}
            for candidate in self.candidates:
                distance = math.sqrt((voter.x - candidate.x) ** 2 + (voter.y - candidate.y) ** 2 + (voter.z - candidate.z) ** 2)
                distances[candidate] = distance         
            sorted_dict = sorted(distances, key = distances.get)
            self.ballots.append(sorted_dict)
        

    def plurality(self):
        votes = {}
        for ballot in self.ballots:
            if ballot[0] in votes:
                votes[ballot[0]] += 1
            else:
                votes[ballot[0]] = 1
        self.sorted_dict = sorted(votes.items(), key = lambda kv: kv[1], reverse = True)      
        return self.sorted_dict[0][0]

    def borda(self):
        points = {}
        for ballot in self.ballots:
            n = len(ballot)
            i = 1
            for candidate in ballot:
                if candidate in points:
                    points[candidate] += (n - i)
                else:
                    points[candidate] = (n - i)
                i += 1
        sorted_dict = sorted(points.items(), key = lambda kv: kv[1], reverse = True)     
        return sorted_dict[0][0]

    def STV(self):
        votes = []
        for ballot in self.ballots:
            vote = Vote(ballot)
            votes.append(vote)
        election = Election(votes)
        election.run_election()
        winner = election.winner
        
        return winner
    
    def head_to_head(self,c_type=0.5):
        points = {}
        score1 = 0
        score2 = 0
        for i in range(len(self.candidates)):
            for j in range(i + 1, len(self.candidates)):
                for ballot in self.ballots:
                    found = False
                    k = 0
                    while not found:
                        if ballot[k] == self.candidates[i]:
                            score1 += 1
                            found = True
                        elif ballot[k] == self.candidates[j]:
                            score2 += 1
                            found = True
                        k += 1            
                if score1 == score2:
                    if self.candidates[i] in points:
                        points[self.candidates[i]] += c_type
                    else:
                        points[self.candidates[i]] = c_type
                    if self.candidates[j] in points:
                        points[self.candidates[j]] += c_type
                    else:
                        points[self.candidates[j]] = c_type
                elif score1 > score2:
                    if self.candidates[i] in points:
                        points[self.candidates[i]] += 1
                    else:
                        points[self.candidates[i]] = 1
                else:
                    if self.candidates[j] in points:
                        points[self.candidates[j]] += 1
                    else:
                        points[self.candidates[j]] = 1
                score1 = 0
                score2 = 0
        sorted_dict = sorted(points.items(), key = lambda kv: kv[1], reverse = True)
        
        #check if a Condorcet Winner exists
        self.hasCondorcetWinner = False
        if sorted_dict[0][1] == len(self.candidates) - 1:
            self.hasCondorcetWinner = True
            self.condorcetWinner = sorted_dict[0][0]
        return sorted_dict
        
    def copeland(self,c_type=0.5):
        sorted_dict = self.head_to_head(c_type)
        return sorted_dict[0][0]

    def minimax(self): #Outputs the greatest θ-winning candidate when k = 1, ie: the minimax voting rule
        #Input: 
        #candidates - an array of each candidate
        #ballots - an array of each ballot, ballots cannot include candidates that do not appear in "candidates"    
        ballots = self.ballots
        candidates = self.candidates
        
        #A1: Initialize variables

        #Greatest θ-winning set variables
        max_coefficient = 0
        greatest_theta_winning_sets = []


        #Number of ballots (n) and candidates (m)
        n = len(ballots)
        m = len(candidates)

        #A2: Directly compute θ coefficient for each pair by iterating through each ballot
        
        for j in range(m):
            counter = 0 #Initialize counter
            min_theta = 100000000 #Initialize min theta
            for k in range(m):
                #To account for datasests where not all candidates are ranked, we need to check if the pair of candidates are in the ballot, every non listed candidate is considered to be of lower rank than any listed candidates
                if k != j: #If k is not j
                    current_theta = 0 #Initialize current theta
                    for ballot in ballots:
                        if candidates[k] not in ballot:
                            if candidates[j] in ballot:
                                counter += 1
                        elif candidates[j] in ballot: #If candidate in the ballot
                            if ballot.index(candidates[j]) < ballot.index(candidates[k]): #If any candidate beats k
                                counter += 1
                    #Normalize θ by the total number of votes to get the coefficient
                    current_theta = counter / n 
                    if current_theta < min_theta:
                        min_theta = current_theta
                    counter = 0 #Reset counter
            this_coefficent = min_theta #Set the pair coefficient to the minimum theta found
            if this_coefficent > max_coefficient:
                max_coefficient = this_coefficent
                greatest_theta_winning_sets = [(candidates[j])]
            elif this_coefficent == max_coefficient and max_coefficient != 0: #If there is a tie, not sure if this would ever happen when k = 1
                greatest_theta_winning_sets.append((candidates[j]))

        #A3: Return greatest θ-winning sets
        return greatest_theta_winning_sets[0]

    def pluralityVeto(self):
        # plurality stage - each candidate is given score equals the number of times they are first-choice
        points = {}
        for ballot in self.ballots:
            if ballot[0] in points:
                points[ballot[0]] += 1
            else:
                points[ballot[0]] = 1
        
        # veto stage 
        numToRemove = len(points) - 1
        while numToRemove>0:
            for ballot in self.ballots:
                # find the bottom-choice candidate among the the standing one
                k = -1
                while not ballot[k] in points:
                    k -= 1

                # decrement score 
                points[ballot[k]] -= 1

                # eleminate the candidate when score reaches 0
                if points[ballot[k]] == 0:
                    points.pop(ballot[k]) 
                    numToRemove -= 1
                    if numToRemove == 0:
                        break
        
        # the last standing candidate is the winner
        winner = list(points)[0]
        return winner

    def getScores(self):
        totalScores = {}
        for voter in self.voters:
            distances = {}
            minDis = float('inf')
            maxDis = 0


            for candidate in self.candidates:
                distance = math.sqrt((voter.x - candidate.x) ** 2 + (voter.y - candidate.y) ** 2 + (voter.z - candidate.z) ** 2)
                distances[candidate] = int(distance)
                if distance >= maxDis:
                    maxDis = int(distance)
                if distance <= minDis:
                    minDis = int(distance)
            disRange = maxDis - minDis
            scale = round(disRange/6)
            scoringMatrix = []
            zero = list(range(maxDis-scale,maxDis+1))
            one = list(range(maxDis-(scale*2),maxDis-scale))
            two = list(range(maxDis-(scale*3),maxDis-(scale*2)))
            three = list(range(maxDis-(scale*4),maxDis-(scale*3)))
            four = list(range(maxDis-(scale*5),maxDis-(scale*4)))
            five = list(range(minDis,maxDis-(scale*5)))
            scoringMatrix.append(zero)
            scoringMatrix.append(one)
            scoringMatrix.append(two)
            scoringMatrix.append(three)
            scoringMatrix.append(four)
            scoringMatrix.append(five)
            for candidate in self.candidates:
                dis = distances[candidate]
                i = 0
                for score in scoringMatrix:
                    if dis in score:
                        distances[candidate] = i
                        if candidate not in totalScores:
                            totalScores[candidate] = i
                        else:
                            totalScores[candidate] += i
                    i += 1
            voter.setScores(distances)
        return totalScores

    def runoff(self, can1,can2):
        can1tot = 0
        can2tot = 0
        for voter in self.voters:
            voterBallot = voter.scores
            score1 = voterBallot[can1]
            score2 = voterBallot[can2]
            if score1 > score2:
                can1tot += 1
            elif score2 > score1:
                can2tot += 1
        if can1tot == can2tot:
            return False
        elif can1tot > can2tot:
            return can1
        else:
            return can2

    def STAR(self):
        finalScores = self.getScores()
        sorted_dict = sorted(finalScores, key = finalScores.get, reverse=True)
        firstCandidate = sorted_dict[0]
        secondCandidate = sorted_dict[1]
        winner = self.runoff(firstCandidate, secondCandidate)
        return winner
    
    def distortion(self,candidate):
        if not candidate:
            return False
        
        sumDistance = 0 
        for voter in self.voters:
            distance = math.sqrt((voter.x - candidate.x) ** 2 + (voter.y - candidate.y) ** 2 + (voter.z - candidate.z)**2)
            sumDistance += distance

    
        distortion = sumDistance / self.minDistance
        return distortion
    
    def majorityCheck(self, candidate):
        if self.sorted_dict[0][1] > len(self.voters)/2:
            if candidate != self.sorted_dict[0][0]:
                return False
            else:
                return True
        return None

    def condorcetCheck(self, candidate):
        self.head_to_head()
        if self.hasCondorcetWinner:
            return candidate == self.condorcetWinner
        else:
            return None
        
def main():
    
    #print(test.pluralityVeto())
    #print(test.condorcetCheck(test.plurality()))
    #print(test.condorcetCheck(test.copeland()))

    m = 20
    n = 200
    for i in range(10):
        test = VoteResult3D(m, n, "1D", "normal")
        winner_mini = test.minimax()
        print(f"Minimax winner {i} is {winner_mini} with a distortion of {test.distortion(winner_mini)}")
        winner_copeland = test.copeland()
        print(f"Copeland winner {i} is {winner_copeland} with a distortion of {test.distortion(winner_copeland)}")

if __name__ == "__main__":  
    main()
    