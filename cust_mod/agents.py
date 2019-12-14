"""Implement Agents and Environments (Chapters 1-2).

The class hierarchies are as follows:

Thing ## A physical object that can exist in an environment
    Agent
        Wumpus
    Dirt
    Wall
    ...

Environment ## An environment holds objects, runs simulations
    XYEnvironment
        VacuumEnvironment
        WumpusEnvironment

An agent program is a callable instance, taking percepts and choosing actions
    SimpleReflexAgentProgram
    ...

EnvGUI ## A window with a graphical representation of the Environment

EnvToolbar ## contains buttons for controlling EnvGUI

EnvCanvas ## Canvas to display the environment of an EnvGUI

TODO: Add `Rule` class

"""

# TO DO:
# Implement grabbing correctly.
# When an object is grabbed, does it still have a location?
# What if it is released?
# What if the grabbed or the grabber is deleted?
# What if the grabber moves?
#
# Speed control in GUI does not have any effect -- fix it.

from utils import distance_squared, turn_heading
from statistics import mean
from ipythonblocks import BlockGrid
from IPython import display as disp
from IPython.display import HTML, display
from time import sleep

import random
import copy
import collections
import numbers


# ______________________________________________________________________________


class Thing:
    """This represents any physical object that can appear in an Environment.
    You subclass Thing to get the things you want. Each thing can have a
    .__name__  slot (used for output only)."""

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def is_alive(self):
        """Things that are 'alive' should return true."""
        return hasattr(self, 'alive') and self.alive

    def show_state(self):
        """Display the agent's internal state. Subclasses should override."""
        print("I don't know how to show_state.")

    def display(self, canvas, x, y, width, height):
        """Display an image of this Thing on the canvas."""
        # Do we need this?
        pass


class Agent(Thing):
    """
    Parameters
    ----------
    program: :type:`function` , default=None
              The program of the agent.

    Attributes
    ----------
    alive: `boolean`
            Boolean indicating if the agent is alive.
    
    bump: `boolean`
            Boolean indicating weather the agent is
            bumping into anything.
    
    holding: `list`
            A list of all the `Things` that the agent
            is holding.
    
    performance: `numbers.Number`
            A number indicating the performance of the agent.

    program: `function` or `lambda`
            A function invoked whenever a new percept arrives.
            returns action
    
    can_grab: ``Not yet Implemented!``
    """

    def __init__(self, program=None):
        self.alive = True
        self.bump = False
        self.holding = []
        self.performance = 0
        if program is None or not isinstance(program, collections.Callable):
            print("Can't find a valid program for {}, falling back to default.".format(
                self.__class__.__name__))

            def program(percept):
                return eval(input('Percept={}; action? '.format(percept)))

        self.program = program

    def can_grab(self, thing):
        """Return True if this agent can grab this thing.
        Override for appropriate subclasses of Agent and Thing."""
        return False


def TraceAgent(agent):
    """Wrap the agent's program to print its input and output. This will let
    you see what the agent is doing in the environment."""
    old_program = agent.program

    def new_program(percept):
        action = old_program(percept)
        print('{} perceives {} and does {}'.format(agent, percept, action))
        return action

    agent.program = new_program
    return agent


# ______________________________________________________________________________


"""
Here's the pseudocode:

1. **function** TABLE-DRIVEN-AGENT(*percept*) **returns** an action
    - **persistent**: 
        - *percepts*, a sequence, initially empty
        - *table*, a table of actions, indexed by percept sequence, initially fully specified.
    - append *percept* to the end of *percepts*
    - *action* <- LOOKUP(*percept*, *table*)
    - **return** *action*
    
"""
def TableDrivenAgentProgram(table):
    """This agent selects an action based on the percept sequence.
    It is practical only for tiny domains.
    To customize it, provide as table a dictionary of all
    {percept_sequence:action} pairs. [Figure 2.7]"""
    
    # So, this is the memory of the agent...
    percepts = []
    
    # This is the program to map from the state
    # to actions based on the persept it recieves.
    # This function is invoked everytime a new
    # percept is recieved.
    def program(percept):
        
        # Add new percept to percept sequence...
        percepts.append(percept)
        
        # Get the coresponding action from the table
        # and return it.
        action = table.get(tuple(percepts))
        return action

    return program


def RandomAgentProgram(actions):
    """An agent that chooses an action at random, ignoring all percepts.
    >>> list = ['Right', 'Left', 'Suck', 'NoOp']
    >>> program = RandomAgentProgram(list)
    >>> agent = Agent(program)
    >>> environment = TrivialVacuumEnvironment()
    >>> environment.add_thing(agent)
    >>> environment.run()
    >>> environment.status == {(1, 0): 'Clean' , (0, 0): 'Clean'}
    True
    """
    return lambda percept: random.choice(actions)


# ______________________________________________________________________________


"""
Here's the pseudocode:

2. **function** SIMPLE-REFLEX-AGENT(*percept*) returns an action
    - **persistent**:
        - *rules*: a set of condition-action rules
    - *state*  <- INTERPRET-INPUT(*percept*)
    - *rule*   <- RULE-MATCH(state, rules)
    - *action* <- rule.ACTION
    - return *action*
    
"""
def SimpleReflexAgentProgram(rules, interpret_input):
    """This agent takes action based solely on the percept. [Figure 2.10]"""
    # Currently not deployed!
    # So, this is the progrm that takes as input a percept
    # and interprets the input as provided by interpret_input.
    # Then it searches for the rule in the rule_book and returns
    # the coresponding action.
    def program(percept):
        state = interpret_input(percept)
        rule = rule_match(state, rules)
        # The implementation of `Rule` class will
        # help deploy the model.
        action = rule.action # Again, Not yet implemented!
        return action

    return program


"""
Here's the pseudocode:

1. **function** MODEL-BASED-REFLEX-AGENT(*percept*) returns *action*
    - **persistent**:
        - *state*, the agent's current conception of the world state
        - *model*, a description of how the next state depends on current state and action
        - *rules*, a rule book containing condition-action rules
        - *action*, the most recent action
    - *state* <- UPDATE-STATE(*state*, *action*, *percept*, *model*)
    - *rule* <- RULE-MATCH(*state*, *rules*)
    - *action* <- rule.ACTION
    - return *action*
        
"""
def ModelBasedReflexAgentProgram(rules,        # A rule book specifing what to do in each state.
                                 update_state, # Function that abstracts a percept using the model.
                                 model         # Encloses how the world evolves.
                                ):
    """This agent takes action based on the percept and state. [Figure 2.12]"""
    # currently not deployed!
    def program(percept):
        program.state = update_state(program.state, program.action, percept, model)
        rule = rule_match(program.state, rules)
        action = rule.action # not implemented yet!
        return action

    # Un-necessary syntactic sugar!
    program.state = program.action = None
    return program


def rule_match(state, rules):
    """Find the first rule that matches state."""

    # No rule class has yet been implemented!
    for rule in rules:
        if rule.matches(state):
            return rule


# ______________________________________________________________________________


loc_A, loc_B = (0, 0), (1, 0)  # The two locations for the Vacuum world


def RandomVacuumAgent():
    """Randomly choose one of the actions from the vacuum environment.
    >>> agent = RandomVacuumAgent()
    >>> environment = TrivialVacuumEnvironment()
    >>> environment.add_thing(agent)
    >>> environment.run()
    >>> environment.status == {(1,0):'Clean' , (0,0) : 'Clean'}
    True
    """
    return Agent(RandomAgentProgram(['Right', 'Left', 'Suck', 'NoOp']))


# TODO: Add examples
def TableDrivenVacuumAgent():
    """[Figure 2.3]"""
    table = {((loc_A, 'Clean'),): 'Right',
             ((loc_A, 'Dirty'),): 'Suck',
             ((loc_B, 'Clean'),): 'Left',
             ((loc_B, 'Dirty'),): 'Suck',
             ((loc_A, 'Dirty'), (loc_A, 'Clean')): 'Right',
             ((loc_A, 'Clean'), (loc_B, 'Dirty')): 'Suck',
             ((loc_B, 'Clean'), (loc_A, 'Dirty')): 'Suck',
             ((loc_B, 'Dirty'), (loc_B, 'Clean')): 'Left',
             ((loc_A, 'Dirty'), (loc_A, 'Clean'), (loc_B, 'Dirty')): 'Suck',
             ((loc_B, 'Dirty'), (loc_B, 'Clean'), (loc_A, 'Dirty')): 'Suck'}
    return Agent(TableDrivenAgentProgram(table))


def ReflexVacuumAgent():
    """A reflex agent for the two-state vacuum environment. [Figure 2.8]
    >>> agent = ReflexVacuumAgent()
    >>> environment = TrivialVacuumEnvironment()
    >>> environment.add_thing(agent)
    >>> environment.run()
    >>> environment.status == {(1,0):'Clean' , (0,0) : 'Clean'}
    True
    """

    def program(percept):
        location, status = percept
        if status == 'Dirty':
            return 'Suck'
        elif location == loc_A:
            return 'Right'
        elif location == loc_B:
            return 'Left'

    return Agent(program)


def ModelBasedVacuumAgent():
    """An agent that keeps track of what locations are clean or dirty.
    >>> agent = ModelBasedVacuumAgent()
    >>> environment = TrivialVacuumEnvironment()
    >>> environment.add_thing(agent)
    >>> environment.run()
    >>> environment.status == {(1,0):'Clean' , (0,0) : 'Clean'}
    True
    """
    model = {loc_A: None, loc_B: None}

    def program(percept):
        """Same as ReflexVacuumAgent, except if everything is clean, do NoOp."""
        location, status = percept
        model[location] = status  # Update the model here
        if model[loc_A] == model[loc_B] == 'Clean':
            return 'NoOp'
        elif status == 'Dirty':
            return 'Suck'
        elif location == loc_A:
            return 'Right'
        elif location == loc_B:
            return 'Left'

    return Agent(program)


# ______________________________________________________________________________

# So, here starts the fucked up part!!!

# Abstrat class
class Environment:
    """
    Abstract class representing an Environment. 'Real' Environment classes
    inherit from this. Your Environment will typically need to implement:
        percept:           Define the percept that an agent sees.
        execute_action:    Define the effects of executing an action.
                           Also update the agent.performance slot.
    The environment keeps a list of .things and .agents (which is a subset
    of .things). Each agent has a .performance slot, initialized to 0.
    Each thing has a .location slot, even though some environments may not
    need this.

    Attributes
    ----------
    things: `list`
            A python list of all the items in the environment.
    
    agents: `list`
            A list of `Agent` objects containing all the
            agents in the environment.
    
    thing_classes: `method`
            List of classes that can go in the environment.
    
    percept: `method` (to be overridden)
            Returns the percept that the agent sees at that point
            in the environment.
    
    execute_action: `method` (to be overridden)
            Execute a action to update the environment. The
            environment doesn't get updated until this
            method is inveoked with either a action or
            a list of actions

    default_location: `method` (to be overridden)
            Returns the location to put a thing in the environment
            if not specified.
    
    exogenous_change: `method` (to be overridden)
            Reset the environment.

    is_done: `method` (to be overridden)
            Returns `True` if all the agents in the
            environment are dead...

    step: `method` (can be overridden)
            Move one time step forward in the environment.
            This method invokes the program of all the agents
            in `agents` list using `percept` method to get the
            percept of the agent and executes the action
            of all the agents one by one.
    
    run: `method` (can be overridden)
            Run the environment for given number of time steps.
    """

    def __init__(self):
        self.things = [] # Wall, Stone
        self.agents = [] # Agents

    def thing_classes(self):
        # OK Honestly really bad documentation!!!
        # "Maybe", it returns a list of agents and things...
        # But, then , why not just return self.things + self.agents??
        # To be overridden
        # NOTE: I dont think we actually need a function for this!
        return []  # List of classes that can go into environment

    def percept(self, agent):
        # To be overridden...
        """Return the percept that the agent sees at this point. (Implement this.)"""
        raise NotImplementedError

    def execute_action(self, agent, action):
        # I think that this return the result of our action.
        # Means return the new percept as a result of our action
        # To be Overridden
        """Change the world to reflect this action. (Implement this.)"""
        raise NotImplementedError

    def default_location(self, thing):
        # I think this method is called when a new thing is added
        # to the environment...
        """Default location to place a new thing with unspecified location."""
        return None

    def exogenous_change(self):
        # ?? Will see later...
        # To be overridden
        """If there is spontaneous change in the world, override this."""
        pass

    def is_done(self):
        # Mostly not overridden!
        """By default, we're done when we can't find a live agent."""
        return not any(agent.is_alive() for agent in self.agents)

    def step(self):
        # Is not overridden if exogenous changes are 
        # not a result of an action taken by the agent...
        """Run the environment for one time step. If the
        actions and exogenous changes are independent, this method will
        do. If there are interactions between them, you'll need to
        override this method."""
        
        # If we are not done
        if not self.is_done():
            actions = [] # start tanking actions.
            for agent in self.agents:
                
                # So, we basically check if the agent is alive.
                # If it is, then append an action that is returned
                # by the program of the agent. The program takes as
                # argument the current percept of the environment...
                # percept method of this class will return the current
                # percept and (i think,) append it to the list of percepts according
                # to the passed agent argument. Here, percept method takes as argument
                # an agent because the percept is generated according to
                # what the agent percieves...
                if agent.alive:
                    actions.append(agent.program(self.percept(agent)))
                else:
                    actions.append("")
            
            # So, now we execute each action one by one and see
            # how the states of the environment change...
            for (agent, action) in zip(self.agents, actions):
                self.execute_action(agent, action)
                
            # Don't know why does this exisits here??
            self.exogenous_change()

    def run(self, steps=1000):
        """Run the Environment for given number of time steps."""
        
        # for given number of steps:
        for step in range(steps):
            # If we are done, return.
            # In a nutshell, this method executes
            # repeatedly until all the agents are dead.
            if self.is_done():
                return
            # Take a step!
            self.step()
    
    def list_things_at(self, location, tclass=Thing):
        """Return all things exactly at a given location."""
        if isinstance(location, numbers.Number):
            return [thing for thing in self.things if thing.location == location and isinstance(thing, tclass)]
        return [thing for thing in self.things if all(x==y for x,y in zip(thing.location, location)) and isinstance(thing, tclass)]

    def some_things_at(self, location, tclass=Thing):
        """Return true if at least one of the things at location
        is an instance of class tclass (or a subclass)."""
        # checks if any thigs are at a given location..
        return self.list_things_at(location, tclass) != []

    def add_thing(self, thing, location=None):
        """Add a thing to the environment, setting its location. For
        convenience, if thing is an agent program we make a new agent
        for it. (Shouldn't need to override this.)"""
        
        # I think this repo has no error handling all in all.
        if not isinstance(thing, Thing):
            thing = Agent(thing)
        
        if thing in self.things:
            print("Can't add the same thing twice")
        else:
            # Set the location of the thing.
            thing.location = location if location is not None else self.default_location(thing)
            
            # Append the thing to the environment.
            self.things.append(thing)
            
            # If the thing is an agent then set its performance to 0
            # and also add it to the agents list.
            # In a nutshell:
            # Object of class Thing is added to things list.
            # Object of agent is added to both the things and agents list.
            if isinstance(thing, Agent):
                thing.performance = 0
                self.agents.append(thing)

    def delete_thing(self, thing):
        """Remove a thing from the environment."""
        try:
            self.things.remove(thing)
        except ValueError as e:
            print(e)
            print("  in Environment delete_thing")
            print("  Thing to be removed: {} at {}".format(thing, thing.location))
            print("  from list: {}".format([(thing, thing.location) for thing in self.things]))
        if thing in self.agents:
            self.agents.remove(thing)


class Direction:
    """A direction class for agents that want to move in a 2D plane
        Usage:
            d = Direction("down")
            To change directions:
            d = d + "right" or d = d + Direction.R #Both do the same thing
            Note that the argument to __add__ must be a string and not a Direction object.
            Also, it (the argument) can only be right or left."""

    R = "right"
    L = "left"
    U = "up"
    D = "down"

    def __init__(self, direction):
        self.direction = direction

    def __add__(self, heading):
        """
        >>> d = Direction('right')
        >>> l1 = d.__add__(Direction.L)
        >>> l2 = d.__add__(Direction.R)
        >>> l1.direction
        'up'
        >>> l2.direction
        'down'
        >>> d = Direction('down')
        >>> l1 = d.__add__('right')
        >>> l2 = d.__add__('left')
        >>> l1.direction == Direction.L
        True
        >>> l2.direction == Direction.R
        True
        """
        
        # If we are heading to the right then
        # we will change the directions as follows:
        # heading = down
        # right = down
        # left = up
        if self.direction == self.R:
            return {
                self.R: Direction(self.D), # "right": Direction("down")
                self.L: Direction(self.U),
            }.get(heading, None)
        # heading = left
        # right = up
        # left = down
        elif self.direction == self.L:
            return {
                self.R: Direction(self.U),
                self.L: Direction(self.D),
            }.get(heading, None)
        # heading = up
        # right = right
        # left = left
        elif self.direction == self.U:
            return {
                self.R: Direction(self.R),
                self.L: Direction(self.L),
            }.get(heading, None)
        # heading = down
        # right = left
        # left = right
        elif self.direction == self.D:
            return {
                self.R: Direction(self.L),
                self.L: Direction(self.R),
            }.get(heading, None)

    def move_forward(self, from_location):
        """
        >>> d = Direction('up')
        >>> l1 = d.move_forward((0, 0))
        >>> l1
        (0, -1)
        >>> d = Direction(Direction.R)
        >>> l1 = d.move_forward((0, 0))
        >>> l1
        (1, 0)
        """
        # So, we are not moving the agent here.
        # We are moving the entire environment.
        # Imagine a camera capturing a particular
        # scene of the environment and then we are moving
        # that camera left, right, up or down tracking the
        # point "from_location".
        iter_class = from_location.__class__
        x, y = from_location
        if self.direction == self.R:
            return iter_class((x + 1, y))
        elif self.direction == self.L:
            return iter_class((x - 1, y))
        elif self.direction == self.U:
            # polarity reversed!
            return iter_class((x, y - 1))
        elif self.direction == self.D:
            return iter_class((x, y + 1))


class XYEnvironment(Environment):
    """This class is for environments on a 2D plane, with locations
    labelled by (x, y) points, either discrete or continuous.

    Agents perceive things within a radius. Each agent in the
    environment has a .location slot which should be a location such
    as (0, 1), and a .holding slot, which should be a list of things
    that are held."""

    def __init__(self, width=10, height=10):
        super().__init__()

        self.width = width
        self.height = height
        self.observers = []
        # Sets iteration start and end (no walls).
        self.x_start, self.y_start = (0, 0)
        self.x_end, self.y_end = (self.width, self.height)
    
    # This distance is the distance upto which the agent can
    # see things...
    perceptible_distance = 1

    def things_near(self, location, radius=None):
        """Return all things within radius of location."""
        if radius is None:
            radius = self.perceptible_distance
            
        # So, we are finding things
        # at euclidean distance less
        # than the redius...
        radius2 = radius * radius
        
        # We are returning a tuple (thing, squared_distance_of_thing_from_agent)
        # if the thing is at a distance less than radius.
        return [(thing, radius2 - distance_squared(location, thing.location))
                for thing in self.things if distance_squared(
                location, thing.location) <= radius2]

    def percept(self, agent):
        """By default, agent perceives things within a default radius."""
        
        # So, a list of things and its distance is the
        # default percept of all agents. This will be mostly
        # overridden as most agents have different percepts...
        return self.things_near(agent.location)

    def execute_action(self, agent, action):
        # Maybe, this method is overidden depending
        # on the actions available in the environment...
        
        # Ok, so agent is not currently bumping any things...
        agent.bump = False
        
        if action == 'TurnRight':
            agent.direction += Direction.R # We turn our head to right instead of moving one step right!
        elif action == 'TurnLeft':
            agent.direction += Direction.L # See the comment above...
        elif action == 'Forward':
            # Take a step forward. move_forward increments the co-ordinates of the agent.
            agent.bump = self.move_to(agent, agent.direction.move_forward(agent.location))
        #         elif action == 'Grab':
        #             things = [thing for thing in self.list_things_at(agent.location)
        #                     if agent.can_grab(thing)]
        #             if things:
        #                 agent.holding.append(things[0])
        elif action == 'Release':
            if agent.holding:
                agent.holding.pop()

    def default_location(self, thing):
        return (random.choice(self.width), random.choice(self.height))

    def move_to(self, thing, destination):
        """Move a thing to a new location. Returns True on success or False if there is an Obstacle. <- No, it doesn't
        If thing is holding anything, they move with him."""
        
        # Returns true if we are bumping something...
        thing.bump = self.some_things_at(destination, Obstacle)
        if not thing.bump:
            thing.location = destination # move the thing to destination
            for o in self.observers:
                o.thing_moved(thing) # update all the observers that the thing moved.
            for t in thing.holding:
                self.delete_thing(t) # delete all the things that the agent is holding.
                self.add_thing(t, destination) # add all the things back at the destination.
                t.location = destination # update the location of the agent.
        # Wait... thing.bump will be true if Obstracle is present at the destination
        # So, we are returning true if an obstracle is present instead of false!!!
        # But its used to update the agent's bump status. So, yeah, it is correctly
        # implemented but the doc comment needs an update...
        return thing.bump

    def add_thing(self, thing, location=(1, 1), exclude_duplicate_class_items=False):
        """Add things to the world. If (exclude_duplicate_class_items) then the item won't be
        added if the location has at least one item of the same class."""
        # Check if the location entered is inbounds.
        if self.is_inbounds(location):
            # If exclude_duplicate_class_items = True then 
            # check if any items we are holding are of the
            # same class as thing and if we find such a thing
            # return...
            if (exclude_duplicate_class_items and
                    any(isinstance(t, thing.__class__) for t in self.list_things_at(location))):
                return
            # Otherwise, add that thing at the location
            super().add_thing(thing, location)

    def is_inbounds(self, location):
        """Checks to make sure that the location is inbounds (within walls if we have walls)"""
        x, y = location
        return not (x < self.x_start or x >= self.x_end or y < self.y_start or y >= self.y_end)

    def random_location_inbounds(self, exclude=None):
        """Returns a random location that is inbounds (within walls if we have walls)"""
        # Not sure why is this useful???
        location = (random.randint(self.x_start, self.x_end),
                    random.randint(self.y_start, self.y_end))
        if exclude is not None:
            while location == exclude:
                location = (random.randint(self.x_start, self.x_end),
                            random.randint(self.y_start, self.y_end))
        return location

    def delete_thing(self, thing):
        """Deletes thing, and everything it is holding (if thing is an agent)"""
        # If thing is an agent, we also want
        # to delete all the things that it
        # is holding...
        if isinstance(thing, Agent):
            for obj in thing.holding:
                super().delete_thing(obj) # delete the thing
                for obs in self.observers:
                    obs.thing_deleted(obj) # update the observer

        super().delete_thing(thing) # delete the thing.
        for obs in self.observers:
            obs.thing_deleted(thing) # update the observer.

    def add_walls(self):
        """Put walls around the entire perimeter of the grid."""
        for x in range(self.width):
            self.add_thing(Wall(), (x, 0))
            self.add_thing(Wall(), (x, self.height - 1))
        for y in range(1, self.height - 1):
            self.add_thing(Wall(), (0, y))
            self.add_thing(Wall(), (self.width - 1, y))

        # Updates iteration start and end (with walls).
        self.x_start, self.y_start = (1, 1)
        self.x_end, self.y_end = (self.width - 1, self.height - 1)

    def add_observer(self, observer):
        """Adds an observer to the list of observers.
        An observer is typically an EnvGUI.

        Each observer is notified of changes in move_to and add_thing,
        by calling the observer's methods thing_moved(thing)
        and thing_added(thing, location)."""
        self.observers.append(observer)

    def turn_heading(self, heading, inc):
        # Don't know anything about this...???
        """Return the heading to the left (inc=+1) or right (inc=-1) of heading."""
        return turn_heading(heading, inc)


class Obstacle(Thing):
    """Something that can cause a bump, preventing an agent from
    moving into the same square it's in."""
    pass


class Wall(Obstacle):
    pass


# ______________________________________________________________________________


class GraphicEnvironment(XYEnvironment):
    def __init__(self, width=10, height=10, boundary=True, color={}, display=False):
        """Define all the usual XYEnvironment characteristics,
        but initialise a BlockGrid for GUI too."""
        super().__init__(width, height) # initialize width and height
        # create a block grid of size height, width and fill it with color (r,g,b) (here (200,200,200))...
        self.grid = BlockGrid(width, height, fill=(200, 200, 200))
        if display:
            self.grid.show() # show the grid on the ipython cell
            self.visible = True
        else:
            self.visible = False
        self.bounded = boundary
        self.colors = color

    def get_world(self):
        """Returns all the items in the world in a format
        understandable by the ipythonblocks BlockGrid."""
        result = []
        x_start, y_start = (0, 0)
        x_end, y_end = self.width, self.height
        for x in range(x_start, x_end):
            row = []
            for y in range(y_start, y_end):
                # Append all the things at location [x, y] for all (x,y) in our grid...
                row.append(self.list_things_at([x, y]))
            # Append all the rows to result
            result.append(row)
        return result

    """
    def run(self, steps=1000, delay=1):
        "" "Run the Environment for given number of time steps,
        but update the GUI too." ""
        for step in range(steps):
            sleep(delay)
            if self.visible:
                self.reveal()
            if self.is_done():
                if self.visible:
                    self.reveal()
                return
            self.step()
        if self.visible:
            self.reveal()
    """

    def run(self, steps=1000, delay=1):
        """Run the Environment for given number of time steps,
        but update the GUI too."""
        for step in range(steps):
            # update simply deletes the currently displaying
            # gui and adds all the new things to the environment
            # and displays it again...
            self.update(delay)
            if self.is_done(): # if we are done then break
                break
            self.step()
        self.update(delay)

    def update(self, delay=1):
        sleep(delay)
        self.reveal()

    def reveal(self):
        """Display the BlockGrid for this world - the last thing to be added
        at a location defines the location color."""
        self.draw_world()
        disp.clear_output(1)
        self.grid.show()
        self.visible = True

    def draw_world(self):
        # Set the color of the grid
        self.grid[:] = (200, 200, 200)
        world = self.get_world() # get the list of things in the world
        for x in range(0, len(world)):
            for y in range(0, len(world[x])):
                if len(world[x][y]):
                    # If the list of things at (x,y) is non-empty
                    # then find the color coresponding to the class name
                    # of the last thing present at (x,y)
                    # and paint the cell (x,y) with the color...
                    self.grid[y, x] = self.colors[world[x][y][-1].__class__.__name__]

    def conceal(self):
        """Hide the BlockGrid for this world"""
        self.visible = False
        display(HTML(''))


# ______________________________________________________________________________
# Continuous environment

class ContinuousWorld(Environment):
    """Model for Continuous World"""

    def __init__(self, width=10, height=10):
        # Not sure, where would this be useful!!
        super().__init__()
        self.width = width
        self.height = height

    def add_obstacle(self, coordinates):
        self.things.append(PolygonObstacle(coordinates))


class PolygonObstacle(Obstacle):

    def __init__(self, coordinates):
        """Coordinates is a list of tuples."""
        super().__init__()
        self.coordinates = coordinates


# ______________________________________________________________________________
# Vacuum environment


class Dirt(Thing):
    pass


class VacuumEnvironment(XYEnvironment):
    """The environment of [Ex. 2.12]. Agent perceives dirty or clean,
    and bump (into obstacle) or not; 2D discrete world of unknown size;
    performance measure is 100 for each dirt cleaned, and -1 for
    each turn taken."""

    def __init__(self, width=10, height=10):
        # So, we initialize the environment normally!
        super().__init__(width, height)
        self.add_walls()

    def thing_classes(self):
        return [Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent,
                TableDrivenVacuumAgent, ModelBasedVacuumAgent]

    def percept(self, agent):
        """The percept is a tuple of ('Dirty' or 'Clean', 'Bump' or 'None').
        Unlike the TrivialVacuumEnvironment, location is NOT perceived."""
        # status isn't a tuple. Its just a string.
        status = ('Dirty' if self.some_things_at(
            agent.location, Dirt) else 'Clean')
        # This variable isn't a tuple... Its just a string.
        bump = ('Bump' if agent.bump else 'None')
        return (status, bump) # return a tuple of strings...

    def execute_action(self, agent, action):
        agent.bump = False
        if action == 'Suck': # We add one more condition of sucking the dirt.
            dirt_list = self.list_things_at(agent.location, Dirt)
            if dirt_list != []:
                dirt = dirt_list[0]
                agent.performance += 100
                self.delete_thing(dirt)
        else:
            super().execute_action(agent, action)

        if action != 'NoOp':
            agent.performance -= 1
    
    def is_done(self):
        return all([not isinstance(thing, Dirt) for thing in self.things]) or not all(agent.is_alive() for agent in self.agents)


class TrivialVacuumEnvironment(Environment):
    """This environment has two locations, A and B. Each can be Dirty
    or Clean. The agent perceives its location and the location's
    status. This serves as an example of how to implement a simple
    Environment."""

    def __init__(self):
        super().__init__()
        self.status = {loc_A: random.choice(['Clean', 'Dirty']),
                       loc_B: random.choice(['Clean', 'Dirty'])}

    def thing_classes(self):
        return [Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent,
                TableDrivenVacuumAgent, ModelBasedVacuumAgent]

    def percept(self, agent):
        """Returns the agent's location, and the location status (Dirty/Clean)."""
        return (agent.location, self.status[agent.location])

    def execute_action(self, agent, action):
        """Change agent's location and/or location's status; track performance.
        Score 10 for each dirt cleaned; -1 for each move."""
        if action == 'Right':
            print("Agent moved to loc_B.")
            agent.location = loc_B
            agent.performance -= 1
        elif action == 'Left':
            print("Agent moved to loc_A.")
            agent.location = loc_A
            agent.performance -= 1
        elif action == 'Suck':
            print("Dirt sucked by agent.")
            if self.status[agent.location] == 'Dirty':
                agent.performance += 10
            self.status[agent.location] = 'Clean'
        elif action == 'NoOp':
            print("No action performed by Agent.")

    def default_location(self, thing):
        """Agents start in either location at random."""
        return random.choice([loc_A, loc_B])
    
    def is_done(self):
        return self.status[loc_A]=='Clean' and self.status[loc_B]=='Clean'


# ______________________________________________________________________________
# The Wumpus World


class Gold(Thing):

    def __eq__(self, rhs):
        """All Gold are equal"""
        return rhs.__class__ == Gold

    pass


class Bump(Thing):
    pass


class Glitter(Thing):
    pass


class Pit(Thing):
    pass


class Breeze(Thing):
    pass


class Arrow(Thing):
    pass


class Scream(Thing):
    pass


class Wumpus(Agent):
    screamed = False
    pass


class Stench(Thing):
    pass


class Explorer(Agent):
    holding = []
    has_arrow = True
    killed_by = ""
    direction = Direction("right")

    def can_grab(self, thing):
        """Explorer can only grab gold"""
        return thing.__class__ == Gold


class WumpusEnvironment(XYEnvironment):
    pit_probability = 0.2  # Probability to spawn a pit in a location. (From Chapter 7.2)

    # Room should be 4x4 grid of rooms. The extra 2 for walls

    def __init__(self, agent_program, width=6, height=6):
        super().__init__(width, height)
        self.init_world(agent_program)

    def init_world(self, program):
        """Spawn items in the world based on probabilities from the book"""

        "WALLS"
        self.add_walls()

        "PITS"
        for x in range(self.x_start, self.x_end):
            for y in range(self.y_start, self.y_end):
                if random.random() < self.pit_probability:
                    self.add_thing(Pit(), (x, y), True)
                    self.add_thing(Breeze(), (x - 1, y), True)
                    self.add_thing(Breeze(), (x, y - 1), True)
                    self.add_thing(Breeze(), (x + 1, y), True)
                    self.add_thing(Breeze(), (x, y + 1), True)

        "WUMPUS"
        w_x, w_y = self.random_location_inbounds(exclude=(1, 1))
        self.add_thing(Wumpus(lambda x: ""), (w_x, w_y), True)
        self.add_thing(Stench(), (w_x - 1, w_y), True)
        self.add_thing(Stench(), (w_x + 1, w_y), True)
        self.add_thing(Stench(), (w_x, w_y - 1), True)
        self.add_thing(Stench(), (w_x, w_y + 1), True)

        "GOLD"
        self.add_thing(Gold(), self.random_location_inbounds(exclude=(1, 1)), True)

        "AGENT"
        self.add_thing(Explorer(program), (1, 1), True)

    def get_world(self, show_walls=True):
        """Return the items in the world"""
        result = []
        x_start, y_start = (0, 0) if show_walls else (1, 1)

        if show_walls:
            x_end, y_end = self.width, self.height
        else:
            x_end, y_end = self.width - 1, self.height - 1

        for x in range(x_start, x_end):
            row = []
            for y in range(y_start, y_end):
                row.append(self.list_things_at((x, y)))
            result.append(row)
        return result

    def percepts_from(self, agent, location, tclass=Thing):
        """Return percepts from a given location,
        and replaces some items with percepts from chapter 7."""
        thing_percepts = {
            Gold: Glitter(),
            Wall: Bump(),
            Wumpus: Stench(),
            Pit: Breeze()}

        """Agents don't need to get their percepts"""
        thing_percepts[agent.__class__] = None

        """Gold only glitters in its cell"""
        if location != agent.location:
            thing_percepts[Gold] = None

        result = [thing_percepts.get(thing.__class__, thing) for thing in self.things
                  if thing.location == location and isinstance(thing, tclass)]
        return result if len(result) else [None]

    def percept(self, agent):
        """Return things in adjacent (not diagonal) cells of the agent.
        Result format: [Left, Right, Up, Down, Center / Current location]"""
        x, y = agent.location
        result = []
        result.append(self.percepts_from(agent, (x - 1, y)))
        result.append(self.percepts_from(agent, (x + 1, y)))
        result.append(self.percepts_from(agent, (x, y - 1)))
        result.append(self.percepts_from(agent, (x, y + 1)))
        result.append(self.percepts_from(agent, (x, y)))

        """The wumpus gives out a loud scream once it's killed."""
        wumpus = [thing for thing in self.things if isinstance(thing, Wumpus)]
        if len(wumpus) and not wumpus[0].alive and not wumpus[0].screamed:
            result[-1].append(Scream())
            wumpus[0].screamed = True

        return result

    def execute_action(self, agent, action):
        """Modify the state of the environment based on the agent's actions.
        Performance score taken directly out of the book."""

        if isinstance(agent, Explorer) and self.in_danger(agent):
            return

        agent.bump = False
        if action == 'TurnRight':
            agent.direction += Direction.R
            agent.performance -= 1
        elif action == 'TurnLeft':
            agent.direction += Direction.L
            agent.performance -= 1
        elif action == 'Forward':
            agent.bump = self.move_to(agent, agent.direction.move_forward(agent.location))
            agent.performance -= 1
        elif action == 'Grab':
            things = [thing for thing in self.list_things_at(agent.location)
                      if agent.can_grab(thing)]
            if len(things):
                print("Grabbing", things[0].__class__.__name__)
                if len(things):
                    agent.holding.append(things[0])
            agent.performance -= 1
        elif action == 'Climb':
            if agent.location == (1, 1):  # Agent can only climb out of (1,1)
                agent.performance += 1000 if Gold() in agent.holding else 0
                self.delete_thing(agent)
        elif action == 'Shoot':
            """The arrow travels straight down the path the agent is facing"""
            if agent.has_arrow:
                arrow_travel = agent.direction.move_forward(agent.location)
                while self.is_inbounds(arrow_travel):
                    wumpus = [thing for thing in self.list_things_at(arrow_travel)
                              if isinstance(thing, Wumpus)]
                    if len(wumpus):
                        wumpus[0].alive = False
                        break
                    arrow_travel = agent.direction.move_forward(agent.location)
                agent.has_arrow = False

    def in_danger(self, agent):
        """Check if Explorer is in danger (Pit or Wumpus), if he is, kill him"""
        for thing in self.list_things_at(agent.location):
            if isinstance(thing, Pit) or (isinstance(thing, Wumpus) and thing.alive):
                agent.alive = False
                agent.performance -= 1000
                agent.killed_by = thing.__class__.__name__
                return True
        return False

    def is_done(self):
        """The game is over when the Explorer is killed
        or if he climbs out of the cave only at (1,1)."""
        explorer = [agent for agent in self.agents if isinstance(agent, Explorer)]
        if len(explorer):
            if explorer[0].alive:
                return False
            else:
                print("Death by {} [-1000].".format(explorer[0].killed_by))
        else:
            print("Explorer climbed out {}.".format("with Gold [+1000]!"
                                                    if Gold() not in self.things else "without Gold [+0]"))
        return True

    # TODO: Arrow needs to be implemented


# ______________________________________________________________________________


def compare_agents(EnvFactory, AgentFactories, n=10, steps=1000):
    """See how well each of several agents do in n instances of an environment.
    Pass in a factory (constructor) for environments, and several for agents.
    Create n instances of the environment, and run each agent in copies of
    each one for steps. Return a list of (agent, average-score) tuples.
    >>> environment = TrivialVacuumEnvironment
    >>> agents = [ModelBasedVacuumAgent, ReflexVacuumAgent]
    >>> result = compare_agents(environment, agents)
    >>> performance_ModelBasedVacuumAgent = result[0][1]
    >>> performance_ReflexVacuumAgent = result[1][1]
    >>> performance_ReflexVacuumAgent <= performance_ModelBasedVacuumAgent
    True
    """
    envs = [EnvFactory() for i in range(n)]
    return [(A, test_agent(A, steps, copy.deepcopy(envs)))
            for A in AgentFactories]


def test_agent(AgentFactory, steps, envs):
    """Return the mean score of running an agent in each of the envs, for steps
    >>> def constant_prog(percept):
    ...     return percept
    ...
    >>> agent = Agent(constant_prog)
    >>> result = agent.program(5)
    >>> result == 5
    True
    """

    def score(env):
        agent = AgentFactory()
        env.add_thing(agent)
        env.run(steps)
        return agent.performance

    return mean(map(score, envs))


# _________________________________________________________________________


__doc__ += """
>>> a = ReflexVacuumAgent()
>>> a.program((loc_A, 'Clean'))
'Right'
>>> a.program((loc_B, 'Clean'))
'Left'
>>> a.program((loc_A, 'Dirty'))
'Suck'
>>> a.program((loc_A, 'Dirty'))
'Suck'

>>> e = TrivialVacuumEnvironment()
>>> e.add_thing(ModelBasedVacuumAgent())
>>> e.run(5)

"""
