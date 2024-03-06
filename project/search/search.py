# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    -construim noi un arbore, pt fiecare nod avem o functie de succesor
    -construim un arbore da mergem in dfs pana ajungem la solutie
    -tre sa vedem vizitatele
    -tre sa implementam o casa cu arbore cred
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    st = Stack()  # Fringe to manage which states to expand
    st.push(problem.getStartState())
    visited = []  # List to check whether state has already been visited
    path = []  # Final direction list
    currentPath = Stack()  # Stack to maintaing path from start to a state
    currentState = st.pop()
    while not problem.isGoalState(currentState):
        if currentState not in visited:
            visited.append(currentState)
            successors = problem.getSuccessors(currentState)
            for child, direction, cost in successors:
                st.push(child)
                tempPath = path + [direction]
                currentPath.push(tempPath)
        currentState = st.pop()
        path = currentPath.pop()
    return path


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    qu = Queue()
    qu.push(problem.getStartState())
    visited = []
    path = []
    currentPath = Queue()
    currentState = qu.pop()
    while not problem.isGoalState(currentState):
        if currentState not in visited:
            visited.append(currentState)
            succ = problem.getSuccessors(currentState)
            for succesor, action, cost in succ:
                temPath = path + [action]
                if succesor not in visited:
                    qu.push(succesor)
                    currentPath.push(temPath)
        currentState = qu.pop()
        path = currentPath.pop()
    return path


def uniformCostSearch(problem: SearchProblem):
    """Search the node of the least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    priorityQ = PriorityQueue()
    priorityQ.push((problem.getStartState(), [], 0), 0)
    visited = []
    path = []
    while not priorityQ.isEmpty():
        state, actions, totalCost = priorityQ.pop()

        if state not in visited:
            visited.append(state)

            if problem.isGoalState(state):
                path = actions
                break
            successors = problem.getSuccessors(state)
            for nextState, action, stepCost in successors:
                if nextState not in visited:
                    nextTotalCost = totalCost + stepCost
                    priorityQ.push((nextState, actions + [action], nextTotalCost), nextTotalCost)
    return path


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    priorityQ = PriorityQueue()  # Fringe to manage which states to expand
    priorityQ.push((problem.getStartState(), [], 0), 0)
    visited = []  # List to check whether state has already been visited
    paths = []  # List to store final sequence of directions
    while not priorityQ.isEmpty():
        state, actions, totalCost = priorityQ.pop()

        if state not in visited:
            visited.append(state)

            if problem.isGoalState(state):
                paths = actions
                break

            successors = problem.getSuccessors(state)
            for nextState, action, stepCost in successors:
                if nextState not in visited:
                    costToNext = totalCost + stepCost + heuristic(nextState, problem)
                    priorityQ.push((nextState, actions + [action], totalCost + stepCost), costToNext)

    return paths


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
