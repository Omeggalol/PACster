# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from __future__ import division

import heapq
import math
import random
from collections import deque

from contest.capture_agents import CaptureAgent
from contest.game import Directions


DIRECTION_STEP = {
    Directions.NORTH: (0, 1),
    Directions.SOUTH: (0, -1),
    Directions.EAST: (1, 0),
    Directions.WEST: (-1, 0),
    Directions.STOP: (0, 0),
}

# Contest loader calls this entry point with two agent indices.
def create_team(first_index, second_index, is_red,
                first='CoordinatedAgent', second='CoordinatedAgent',
                num_training=0, strategy=None):
    """One private communication dictionary per team, with validated options."""
    selected = strategy or CoordinatedAgent.DEFAULT_STRATEGY
    if selected not in CoordinatedAgent.STRATEGIES:
        raise ValueError('Unknown strategy: ' + str(selected))
    classes = {'CoordinatedAgent': CoordinatedAgent,
               'TacticalAgent': TacticalAgent,
               'CoreTacticalAgent': CoreTacticalAgent}
    if first not in classes or second not in classes:
        raise ValueError('Unknown agent class')
    memory = {}
    team = [classes[first](first_index), classes[second](second_index)]
    for agent in team:
        agent.strategy = selected
        agent._team_memory = memory
    return team


class CoreTacticalAgent(CaptureAgent):
    """Coordinated hybrid agent with dynamic objectives and risk-aware routing.

    Key ideas:
      * Two attackers when safe; defend immediately if an enemy invades.
      * Return food before a dangerous encounter; prioritize winning score.
      * Dijkstra costs reflect enemy ghost proximity and dead-end exposure.
      * Noisy-distance opponent beliefs and eaten-food incident tracking.
      * Target persistence, teammate separation, and anti-loop penalties.
    All runtime logic uses stdlib only.
    """

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.start = self._cell(game_state.get_agent_position(self.index))
        self.team = list(self.get_team(game_state))
        self.other = next(i for i in self.team if i != self.index)
        self.enemies = list(self.get_opponents(game_state))
        self.primary = self.index == min(self.team)
        walls = game_state.get_walls()
        self.width, self.height = walls.width, walls.height
        self.home_x = self.width // 2 - 1 if self.red else self.width // 2
        self.enemy_x = self.width // 2 if self.red else self.width // 2 - 1
        self.legal_cells = [(x, y) for x in range(self.width)
                            for y in range(self.height) if not walls[x][y]]
        self.graph = {}
        for x, y in self.legal_cells:
            self.graph[(x, y)] = [(x + dx, y + dy)
                                  for dx, dy in DIRECTION_STEP.values()
                                  if (dx or dy) and 0 <= x + dx < self.width
                                  and 0 <= y + dy < self.height
                                  and not walls[x + dx][y + dy]]
        self.home_portals = [p for p in self.legal_cells if p[0] == self.home_x
                             and (self.enemy_x, p[1]) in self.graph]
        self.enemy_portals = [(self.enemy_x, p[1]) for p in self.home_portals]
        self.trap_depth = self._dead_end_depth()
        self.recent = deque(maxlen=10)
        self.goal = None
        self.goal_kind = None
        self.last_mode = None
        self.last_turn = None
        self.last_known_ghosts = {}
        self.last_defended_food = set(self.get_food_you_are_defending(game_state).as_list())
        self.last_defended_capsules = set(self.get_capsules_you_are_defending(game_state))
        self.incident = None
        self.incident_t = -9999
        self.beliefs = {i: {self._cell(game_state.get_initial_agent_position(i)): 1.0}
                        for i in self.enemies}
        self.belief_tick = None
        # Bound computation time. Framework precalculates get_maze_distance in setup.
        self._rng = random.Random(self.index * 7919 + self.width * 37 + self.height)

    @staticmethod
    def _cell(p):
        return None if p is None else (int(round(p[0])), int(round(p[1])))

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _home(self, p):
        return p is not None and ((p[0] <= self.home_x) if self.red
                                  else (p[0] >= self.home_x))

    def _enemy_home(self, p):
        return p is not None and not self._home(p)

    def _maze(self, a, b):
        if a is None or b is None:
            return 999
        # All calls use integer-grid locations.
        try:
            return self.get_maze_distance(a, b)
        except (KeyError, IndexError, ValueError):
            return self._manhattan(a, b)

    def _dead_end_depth(self):
        """Prune branches to estimate irreversible depth in dead-end corridors."""
        degree = {p: len(adj) for p, adj in self.graph.items()}
        q = deque(p for p, d in degree.items() if d <= 1)
        depth = {p: 0 for p in self.graph}
        removed = set()
        while q:
            p = q.popleft()
            if p in removed:
                continue
            removed.add(p)
            for n in self.graph[p]:
                if n not in removed:
                    depth[n] = max(depth[n], depth[p] + 1)
                    degree[n] -= 1
                    if degree[n] == 1:
                        q.append(n)
        return {p: depth[p] for p in removed}

    def _update_food_incident(self, state, tick):
        defended = set(self.get_food_you_are_defending(state).as_list())
        capsules = set(self.get_capsules_you_are_defending(state))
        newly_missing = (self.last_defended_food - defended) | (self.last_defended_capsules - capsules)
        if newly_missing:
            pos = self._cell(state.get_agent_position(self.index))
            self.incident = min(newly_missing, key=lambda p: self._maze(pos, p))
            self.incident_t = tick
        self.last_defended_food = defended
        self.last_defended_capsules = capsules

    def _update_beliefs(self, state, pos, tick):
        """Lightweight hidden-opponent filter using sonar and team sight radius.

        Avoids pretending that a noisy sonar observation pinpoints an enemy.
        """
        mate = self._cell(state.get_agent_position(self.other))
        sonar = state.get_agent_distances()
        for i in self.enemies:
            direct = self._cell(state.get_agent_position(i))
            if direct is not None:
                self.beliefs[i] = {direct: 1.0}
                continue
            prior = self.beliefs.get(i, {})
            if not prior:
                prior = {p: 1.0 for p in self.graph}
            # Roughly one enemy action occurs between full rounds of observations.
            if self.belief_tick is not None and tick - self.belief_tick >= 2:
                steps = min(3, max(1, (tick - self.belief_tick) // 4))
                for _ in range(steps):
                    new = {}
                    for p, prob in prior.items():
                        adj = self.graph.get(p, ())
                        options = list(adj) + [p]
                        share = prob / max(1, len(options))
                        for nxt in options:
                            new[nxt] = new.get(nxt, 0.0) + share
                    prior = new
            enemy_is_pacman = bool(state.get_agent_state(i).is_pacman)
            reading = sonar[i] if sonar is not None and len(sonar) > i else None
            def allowed(p):
                if self._home(p) != enemy_is_pacman:
                    return False
                if self._manhattan(pos, p) <= 5 or (mate and self._manhattan(mate, p) <= 5):
                    return False
                if reading is not None and abs(self._manhattan(pos, p) - reading) > 6:
                    return False
                return True
            posterior = {p: prob for p, prob in prior.items() if allowed(p)}
            if not posterior:
                posterior = {p: 1.0 for p in self.legal_cells if allowed(p)}
            if posterior:
                # Cap state size so turns remain fast even in large mazes.
                top = sorted(posterior.items(), key=lambda e: e[1], reverse=True)[:160]
                total = sum(v for _, v in top)
                self.beliefs[i] = {p: v / total for p, v in top}
        self.belief_tick = tick

    def _enemy_threat(self, state, tick):
        """Current visible ghosts and short-lived last-seen ghost positions.

        Last-seen memory avoids rapid policy reversal at the five-square
        visibility boundary. Positions are estimates, never claimed exact.
        """
        threats = []
        invaders = []
        for i in self.enemies:
            es = state.get_agent_state(i)
            p = self._cell(state.get_agent_position(i))
            if es.is_pacman:
                if p is not None:
                    invaders.append((i, p, es))
            elif p is not None:
                self.last_known_ghosts[i] = (p, tick)
                threats.append((i, p, es))
            elif i in self.last_known_ghosts:
                old_pos, old_tick = self.last_known_ghosts[i]
                if tick - old_tick <= 24:
                    threats.append((i, old_pos, es))
        return threats, invaders

    def _danger_map(self, threats, carrying, scared_self=0):
        """Per-tile cost of dying before safely banking points."""
        risk = {}
        live = [(p, st) for _, p, st in threats if st.scared_timer <= 2]
        if live:
            # Distance maps for visible defenders; BFS is exact on maze.
            fields = []
            for origin, _ in live:
                dmap = {origin: 0}
                todo = deque([origin])
                while todo:
                    p = todo.popleft()
                    if dmap[p] >= 6:
                        continue
                    for n in self.graph[p]:
                        if n not in dmap:
                            dmap[n] = dmap[p] + 1
                            todo.append(n)
                fields.append(dmap)
            scale = min(2.4, 1.0 + 0.13 * carrying)
            for p in self.legal_cells:
                if self._home(p):
                    continue
                d = min((f.get(p, 99) for f in fields), default=99)
                hazard = {0: 160.0, 1: 65.0, 2: 16.0, 3: 5.0,
                          4: 1.7, 5: 0.45}.get(d, 0.0)
                if hazard:
                    depth = min(8, self.trap_depth.get(p, 0))
                    risk[p] = scale * hazard * (1.0 + 0.10 * depth)
        return risk

    def _routes(self, start, risk, dangerous_when_scared=False):
        """Single-source Dijkstra: path costs and first moves to every target."""
        shortest = {start: 0.0}
        first_action = {start: Directions.STOP}
        todo = [(0.0, 0, start)]
        order = 0
        directions = ((Directions.NORTH, (0, 1)),
                      (Directions.SOUTH, (0, -1)),
                      (Directions.EAST, (1, 0)),
                      (Directions.WEST, (-1, 0)))
        while todo:
            cost, _, p = heapq.heappop(todo)
            if cost > shortest.get(p, float('inf')) + 1e-8:
                continue
            for direction, (dx, dy) in directions:
                n = (p[0] + dx, p[1] + dy)
                if n not in self.graph:
                    continue
                # A small tie-break makes shortest routes favor open junctions.
                step = 1.0 + risk.get(n, 0.0)
                if dangerous_when_scared and self._home(n):
                    step += 0.3
                if step + cost + 1e-8 < shortest.get(n, float('inf')):
                    shortest[n] = step + cost
                    first_action[n] = direction if p == start else first_action[p]
                    order += 1
                    heapq.heappush(todo, (shortest[n], order, n))
        return shortest, first_action

    def _bank_distance(self, pos):
        if self._home(pos):
            return 0
        return min((self._maze(pos, p) for p in self.home_portals), default=999)

    def _best_food(self, state, pos, costs, food, carrying, mate, live_threats):
        """Choose travel-efficient food clusters, not simply the nearest dot."""
        result = None
        best = -1e10
        for f in food:
            cost = costs.get(f, 9999)
            if cost >= 9999:
                continue
            # Nearby dots make the journey useful beyond a single pellet.
            cluster = sum(1 for q in food if self._manhattan(f, q) <= 3)
            efficiency = 2.25 * min(cluster, 5)
            exit_dist = self._bank_distance(f)
            value = 13.0 - cost + efficiency - 0.13 * exit_dist
            # Coordination: avoid routing both agents to the same cluster.
            if mate and self._enemy_home(mate):
                mate_dist = self._maze(mate, f)
                if mate_dist + 1 < self._maze(pos, f):
                    value -= max(0, 6 - mate_dist) * 1.35
            if f == self.goal and self.goal_kind == 'food':
                # Strong hysteresis: protect against rapid target switching
                # at the boundary where an enemy appears/disappears.
                value += 12.0
            if self.recent.count(f) >= 2:
                value -= 3.0
            # Don't enter exposed dead ends if an active ghost can intercept.
            if live_threats and self.trap_depth.get(f, 0) > 2:
                nearest_enemy = min(self._maze(f, g) for g in live_threats)
                value -= max(0, 7 - nearest_enemy) * 1.15
            if value > best:
                best, result = value, f
        return result, best

    def _choose_attack(self, state, pos, legal, threats, carrying, mate):
        food = self.get_food(state).as_list()
        capsules = [self._cell(p) for p in self.get_capsules(state)]
        active = [p for _, p, es in threats if es.scared_timer <= 2]
        scared_active = [p for _, p, es in threats if es.scared_timer > 2]
        threat_dist = min((self._maze(pos, p) for p in active), default=999)
        risk = self._danger_map(threats, carrying)
        costs, path = self._routes(pos, risk)
        score = self.get_score(state)
        remaining = state.data.timeleft if hasattr(state.data, 'timeleft') else 400
        exit_dist = self._bank_distance(pos)
        # Deposit cargo based on level of exposure and remaining game time.
        # Once a bank trip starts, finish it. Otherwise risk conditions may
        # fluctuate one tile later, causing a damaging food/return oscillation.
        return_now = carrying > 0 and self.goal_kind == 'return'
        if carrying:
            return_now = return_now or (carrying >= 8 or
                          (carrying >= 5 and (threat_dist <= 8 or score >= 6)) or
                          (carrying >= 3 and threat_dist <= 5) or
                          (carrying >= 1 and threat_dist <= 3) or
                          len(food) <= 2 or
                          remaining < (exit_dist + 10) * 4 or
                          (score < 0 and remaining < 65 and carrying + score > 0))
        # Bank near game end rather than collecting points that cannot be cashed.
        if carrying and remaining < (exit_dist + 7) * 4:
            return_now = True
        if return_now or len(food) <= 2:
            choices = [(costs[p], p) for p in self.home_portals if p in costs]
            if choices:
                _, dest = min(choices)
                return path[dest], dest, 'return'
        best_goal, best_value = self._best_food(state, pos, costs, food, carrying, mate, active)
        kind = 'food'
        # Capsules are disproportionately useful when close to active ghosts.
        for capsule in capsules:
            d = costs.get(capsule, 9999)
            if d >= 9999:
                continue
            ghost_near_capsule = min((self._maze(capsule, p) for p in active), default=999)
            capsule_value = 9.0 - d
            if active:
                capsule_value += 10.0 if min(threat_dist, ghost_near_capsule) <= 7 else 4.0
            if carrying >= 2:
                capsule_value += 2.0
            if self.goal_kind == 'capsule' and capsule == self.goal:
                capsule_value += 2.5
            if capsule_value > best_value:
                best_value, best_goal, kind = capsule_value, capsule, 'capsule'
        if best_goal is None:
            if self.home_portals:
                best_goal = min(self.home_portals, key=lambda p: costs.get(p, 9999))
                kind = 'return'
            else:
                return self._rng.choice(legal), None, 'fallback'
        action = path.get(best_goal, Directions.STOP)
        # If already on a target, replan toward a different food/portal next turn.
        if best_goal == pos:
            viable = [a for a in legal if a != Directions.STOP]
            if viable:
                action = min(viable, key=lambda a: self._local_action_penalty(pos, a, risk))
        return action, best_goal, kind

    def _local_action_penalty(self, pos, action, risk):
        dx, dy = DIRECTION_STEP[action]
        nxt = (pos[0] + dx, pos[1] + dy)
        return risk.get(nxt, 0.0) + 2.5 * self.recent.count(nxt)

    def _defense_goal(self, state, pos, invaders, tick):
        if invaders:
            # Chase most dangerous invader: cargo and proximity matter.
            scores = []
            for idx, enemy_pos, enemy_state in invaders:
                distance = self._maze(pos, enemy_pos)
                priority = distance - min(8, enemy_state.num_carrying) * 0.9
                scores.append((priority, enemy_pos))
            return min(scores)[1], 'intercept'
        if self.incident is not None and tick - self.incident_t < 28:
            return self.incident, 'investigate'
        # Enemy in our territory but hidden: use filtered belief distribution.
        hidden_targets = []
        for i in self.enemies:
            if not state.get_agent_state(i).is_pacman:
                continue
            for p, pr in self.beliefs.get(i, {}).items():
                if self._home(p):
                    # Balance probability with distance for practical intercepts.
                    hidden_targets.append((pr / (self._maze(pos, p) + 4), p))
        if hidden_targets:
            return max(hidden_targets)[1], 'search'
        # Sentry prefers central choke points near opponent-facing boundary.
        if self.home_portals:
            defended_food = self.get_food_you_are_defending(state).as_list()
            if defended_food:
                target_y = sum(p[1] for p in defended_food) / len(defended_food)
            else:
                target_y = self.height / 2
            return min(self.home_portals, key=lambda p: abs(p[1] - target_y)), 'patrol'
        return self.start, 'patrol'

    def _choose_defense(self, state, pos, legal, threats, invaders, tick):
        own = state.get_agent_state(self.index)
        goal, kind = self._defense_goal(state, pos, invaders, tick)
        mate = self._cell(state.get_agent_position(self.other))
        # If frightened, stay out of melee. Scared ghosts can be eaten.
        if own.scared_timer > 0 and invaders:
            dangerous = [e for _, e, _ in invaders]
            safe_moves = []
            for a in legal:
                dx, dy = DIRECTION_STEP[a]
                nxt = (pos[0] + dx, pos[1] + dy)
                if nxt not in self.graph:
                    continue
                clearance = min(self._maze(nxt, e) for e in dangerous)
                if clearance >= 2:
                    safe_moves.append((min(clearance, 5) * 3 - self._maze(nxt, goal) * .2, a))
            if safe_moves:
                return max(safe_moves)[1], goal, 'scared-defense'
        costs, first = self._routes(pos, {})
        action = first.get(goal, Directions.STOP)
        if goal == pos:
            # Avoid standing completely still when patroling; stay near border.
            if kind == 'patrol':
                options = []
                for a in legal:
                    dx, dy = DIRECTION_STEP[a]
                    nxt = (pos[0] + dx, pos[1] + dy)
                    if self._home(nxt) and nxt in self.graph:
                        alternatives = min((self._maze(nxt, x) for x in self.home_portals), default=10)
                        options.append((-alternatives - self.recent.count(nxt), a))
                if options:
                    action = max(options)[1]
            else:
                action = Directions.STOP
        return action, goal, kind

    def _should_defend(self, state, pos, invaders, tick):
        # Critical: at least one defender when visible/hidden invader is present.
        invading = [i for i in self.enemies if state.get_agent_state(i).is_pacman]
        if invading:
            if len(invading) >= 2:
                return True if not self.primary else (len(invaders) >= 2)
            if self.primary:
                # Primary attacker defends if closer and secondary far/loaded.
                mate = self._cell(state.get_agent_position(self.other))
                here = min((self._maze(pos, p) for _, p, _ in invaders), default=999)
                there = min((self._maze(mate, p) for _, p, _ in invaders), default=999)
                return (here + 3 < there and here < 7)
            return True
        if self.incident and tick - self.incident_t < 16:
            return not self.primary
        # When winning late, guard the lead. Otherwise both agents can gather.
        score = self.get_score(state)
        left = getattr(state.data, 'timeleft', 400)
        if not self.primary and score > 0 and left < 200:
            return True
        return False

    def choose_action(self, game_state):
        legal = game_state.get_legal_actions(self.index)
        if not legal:
            return Directions.STOP
        pos = self._cell(game_state.get_agent_position(self.index))
        if pos is None or pos not in self.graph:
            return legal[0]
        tick = 1200 - getattr(game_state.data, 'timeleft', 1200)
        if self.last_turn is not None and tick <= self.last_turn:
            # The same module may be reused between games; reset transient state.
            self.recent.clear()
            self.goal = None
            self.incident = None
        self.last_turn = tick
        self.recent.append(pos)
        self._update_food_incident(game_state, tick)
        self._update_beliefs(game_state, pos, tick)
        threats, invaders = self._enemy_threat(game_state, tick)
        mate = self._cell(game_state.get_agent_position(self.other))
        carried = game_state.get_agent_state(self.index).num_carrying
        if self._should_defend(game_state, pos, invaders, tick):
            action, goal, kind = self._choose_defense(game_state, pos, legal, threats, invaders, tick)
            self.last_mode = 'defend'
        else:
            action, goal, kind = self._choose_attack(game_state, pos, legal, threats, carried, mate)
            self.last_mode = 'attack'
        self.goal, self.goal_kind = goal, kind
        if action not in legal:
            # Safe fallback from unexpected grid/engine quirks.
            action = min(legal, key=lambda a: self._local_action_penalty(pos, a, {}))
        return action


class TacticalAgent(CoreTacticalAgent):
    """Experimental v3: situation-aware strategic choices with opponent profiles.

    No assumptions about hidden positions being known. Pattern classifications
    are evidence-weighted, deliberately low-confidence, and decay over time.
    """

    ENABLE_PROFILE = True
    ENABLE_HIDDEN_RISK = True
    ENABLE_INTERCEPT = False
    ENABLE_ADAPTIVE_BANK = True
    ENABLE_ESCAPE = False
    ENABLE_META_BANK = False
    # Competition default: v3_selected, winner of local ablation comparisons.
    # Experimental features above can be toggled, but only stable subset ships on.

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.enemy_profiles = {i: {'camp': 0., 'chase': 0., 'raid': 0.,
                                   'retreat': 0., 'samples': 0,
                                   'last_position': None, 'last_tick': None,
                                   'last_carrying': 0, 'last_pac': False,
                                   'entry_rows': {}, 'seen_ghosts': 0}
                               for i in self.enemies}
        self.opponent_mode = {i: 'unknown' for i in self.enemies}
        self.last_threat_dist = 999
        self.return_lock = 0
        self.invader_ids_seen = set()
        self.multi_raider = False

    def _update_beliefs(self, state, pos, tick):
        super()._update_beliefs(state, pos, tick)
        if self.ENABLE_PROFILE:
            self._update_profiles(state, pos, tick)
        # Evidence that a defended pellet vanished means an invader was at
        # that location recently. We do not assume which enemy did it.
        if self.ENABLE_PROFILE and self.incident is not None and tick - self.incident_t < 9:
            candidates = [i for i in self.enemies if state.get_agent_state(i).is_pacman
                          and state.get_agent_position(i) is None]
            if len(candidates) == 1:
                enemy = candidates[0]
                prior = self.beliefs.get(enemy, {})
                boosted = {}
                for p, probability in prior.items():
                    d = self._manhattan(p, self.incident)
                    boosted[p] = probability * (2.5 if d <= 2 else 1.25 if d <= 5 else .65)
                norm = sum(boosted.values())
                if norm:
                    self.beliefs[enemy] = {p: v / norm for p, v in boosted.items()}

    def _update_profiles(self, state, pos, tick):
        for i in self.enemies:
            profile = self.enemy_profiles[i]
            # Decay old evidence: opponents may change strategies mid-game.
            last_t = profile['last_tick']
            if last_t is not None and tick > last_t:
                factor = .985 ** min(32, tick - last_t)
                for name in ('camp', 'chase', 'raid', 'retreat'):
                    profile[name] *= factor
            enemy = self._cell(state.get_agent_position(i))
            es = state.get_agent_state(i)
            if enemy is None:
                continue
            if enemy == profile['last_position'] and last_t == tick:
                continue
            previous = profile['last_position']
            elapsed = tick - last_t if last_t is not None else 999
            profile['samples'] += 1
            if es.is_pacman:
                # The invader choosing deeper home territory or returning
                # to the enemy boundary can be distinguished by movement.
                profile['raid'] += .40
                if previous is not None and elapsed <= 12:
                    dprev = min((self._maze(previous, portal) for portal in self.home_portals), default=999)
                    dnow = min((self._maze(enemy, portal) for portal in self.home_portals), default=999)
                    if dnow < dprev and es.num_carrying >= 2:
                        profile['retreat'] += 1.4
                    elif dnow > dprev:
                        profile['raid'] += .85
            else:
                profile['seen_ghosts'] += 1
                # Stationary near the border with a good line to crossing:
                # probable portal sentry, not an aggressive chase.
                border_dist = min((self._maze(enemy, p) for p in self.enemy_portals), default=999)
                if border_dist <= 3:
                    profile['camp'] += .45
                    if previous == enemy and elapsed <= 12:
                        profile['camp'] += 2.2
                if previous is not None and elapsed <= 12:
                    own_d = self._maze(pos, enemy)
                    prev_d = self._maze(pos, previous)
                    if self._enemy_home(pos) and own_d + 1 < prev_d:
                        profile['chase'] += 1.7
                    if self._enemy_home(pos) and own_d < 5 and border_dist > 3:
                        profile['chase'] += .7
            profile['last_position'] = enemy
            profile['last_tick'] = tick
            profile['last_carrying'] = es.num_carrying
            profile['last_pac'] = bool(es.is_pacman)
            if previous is not None and not self._home(previous) and self._home(enemy):
                row = enemy[1]
                profile['entry_rows'][row] = profile['entry_rows'].get(row, 0) + 1
            effective_samples = profile['samples']
            dominant = max(('camp', 'chase', 'raid', 'retreat'),
                           key=lambda name: profile[name])
            dominant_score = profile[dominant]
            if effective_samples < 3 or dominant_score < 3.0:
                self.opponent_mode[i] = 'unknown'
            else:
                self.opponent_mode[i] = dominant

    def _danger_map(self, threats, carrying, scared_self=0):
        danger = super()._danger_map(threats, carrying, scared_self)
        if not self.ENABLE_HIDDEN_RISK:
            return danger
        # Hidden opponent distribution, but only concentrated beliefs should
        # alter routes much. Never hallucinate a precise hidden location.
        known = {i for i, _, _ in threats}
        for idx, distribution in self.beliefs.items():
            if idx in known or not distribution:
                continue
            # Beliefs for hidden opponents can span most of the board. We add
            # weak expected ghost proximity cost, rather than treating every
            # possible square as a known blocker.
            tops = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:35]
            if not tops:
                continue
            concentration = sum(pr for _, pr in tops)
            if concentration < .20:
                continue
            for origin, likelihood in tops:
                if self._home(origin):
                    continue
                impact = min(.32, likelihood) * (6 + min(12, carrying * 1.5))
                for p in (origin,) + tuple(self.graph.get(origin, ())):
                    if not self._home(p):
                        danger[p] = danger.get(p, 0.0) + impact * (1 if p == origin else .37)
        return danger

    def _defense_goal(self, state, pos, invaders, tick):
        if self.ENABLE_INTERCEPT and invaders and self.home_portals:
            # Intercept high-value invaders at likely escape chokepoints when
            # we can reach the portal before them. Otherwise chase normally.
            best = None
            for _, enemy_pos, es in invaders:
                if es.num_carrying < 3:
                    continue
                portal = min(self.home_portals, key=lambda p: self._maze(enemy_pos, p))
                ed = self._maze(enemy_pos, portal)
                md = self._maze(pos, portal)
                if ed <= 7 and md <= ed + 1:
                    advantage = es.num_carrying * 1.5 - md
                    if best is None or advantage > best[0]:
                        best = (advantage, portal)
            if best:
                return best[1], 'portal-intercept'
        # If no invader is visible, a recently visited entry corridor is a
        # better fallback than the arbitrary board center.
        if self.ENABLE_PROFILE and self.incident is None:
            hidden_invaders = [i for i in self.enemies if state.get_agent_state(i).is_pacman
                               and state.get_agent_position(i) is None]
            if hidden_invaders:
                hist = {}
                for i in hidden_invaders:
                    for y, n in self.enemy_profiles[i]['entry_rows'].items():
                        hist[y] = hist.get(y, 0) + n
                if hist and self.home_portals:
                    favorite = max(hist, key=hist.get)
                    if hist[favorite] >= 2:
                        portal = min(self.home_portals, key=lambda p: abs(p[1] - favorite))
                        return portal, 'pattern-patrol'
        return super()._defense_goal(state, pos, invaders, tick)

    def _should_defend(self, state, pos, invaders, tick):
        # Opponent-policy selector: the aggressive counter-strategy is only
        # enabled once both enemy agents have demonstrated willingness to raid.
        for i in self.enemies:
            if state.get_agent_state(i).is_pacman:
                self.invader_ids_seen.add(i)
        self.multi_raider = len(self.invader_ids_seen) == len(self.enemies)
        if self.ENABLE_META_BANK and not self.multi_raider:
            return CoreTacticalAgent._should_defend(self, state, pos, invaders, tick)
        if not self.ENABLE_ADAPTIVE_BANK:
            return super()._should_defend(state, pos, invaders, tick)
        own = state.get_agent_state(self.index)
        carrying = own.num_carrying
        enemy_has_invaded = [i for i in self.enemies if state.get_agent_state(i).is_pacman]
        # Once one of our attackers has a valuable haul abroad, defend only
        # after banking. Otherwise they waste many turns and risk their loot.
        if carrying >= 3 and self._enemy_home(pos):
            return False
        if not enemy_has_invaded:
            return super()._should_defend(state, pos, invaders, tick)
        score = self.get_score(state)
        left = getattr(state.data, 'timeleft', 400)
        mate = self._cell(state.get_agent_position(self.other))
        # Defending a late lead is worth more than racing for marginal dots.
        if left <= 200 and score >= 4:
            return (not self.primary) or (len(enemy_has_invaded) >= 2 and self._home(pos))
        if not self.primary:
            if carrying >= 2 and self._enemy_home(pos):
                return False
            # Defender far away and teammate near the burglar: let teammate
            # make a cheaper interception if at all possible.
            if invaders and mate and self._home(mate):
                d_me = min(self._maze(pos, p) for _, p, _ in invaders)
                d_mate = min(self._maze(mate, p) for _, p, _ in invaders)
                if d_mate + 4 < d_me and d_me > 10:
                    return False
            return True
        # Primary attacks unless already home and immediately able to catch
        # an intruder with valuable cargo.
        if invaders and self._home(pos) and carrying == 0:
            priority = min((self._maze(pos, p) - min(6, es.num_carrying) * .6
                            for _, p, es in invaders), default=999)
            if priority <= 2 and (mate is None or
                    min(self._maze(mate, p) for _, p, _ in invaders) >= priority + 2):
                return True
        return False

    def _choose_attack(self, state, pos, legal, threats, carrying, mate):
        action, goal, kind = super()._choose_attack(state, pos, legal, threats, carrying, mate)
        if self.ENABLE_ADAPTIVE_BANK and (not self.ENABLE_META_BANK or self.multi_raider) and self._enemy_home(pos) and carrying:
            # Do not overwrite an already locked return plan. Change only
            # situations where an observable pursuer makes the plan risky.
            active = [(i, p) for i, p, st in threats if st.scared_timer <= 2]
            aggressive = [p for i, p in active if self.opponent_mode.get(i) == 'chase']
            distance = min((self._maze(pos, p) for p in aggressive), default=999)
            left = getattr(state.data, 'timeleft', 400)
            score = self.get_score(state)
            more_value = bool(carrying >= 2 and distance <= 8) or (left < 100 and score > 0)
            if kind != 'return' and more_value:
                risk = self._danger_map(threats, carrying)
                costs, first = self._routes(pos, risk)
                if self.home_portals:
                    portal = min(self.home_portals, key=lambda p: costs.get(p, 9999))
                    if portal in first:
                        return first[portal], portal, 'return'
        if self.ENABLE_ESCAPE and self._enemy_home(pos):
            action = self._escape_override(state, pos, legal, threats, carrying, action, goal, kind)
        return action, goal, kind

    def _escape_override(self, state, pos, legal, threats, carrying, original, goal, kind):
        active = [(p, s) for _, p, s in threats if s.scared_timer <= 2]
        if not active:
            return original
        nearest = min(self._maze(pos, p) for p, _ in active)
        if nearest > 3:
            return original
        # Robust one-step game-theoretic avoidance: evaluate the defender's
        # best *legal* reply after each of our candidate moves. A heuristic
        # tiebreaker rewards ongoing progress to the chosen strategic goal.
        # Only activate if the originally chosen move has high immediate risk.
        options = []
        for a in legal:
            delta = DIRECTION_STEP.get(a, (0, 0))
            candidate = (pos[0] + delta[0], pos[1] + delta[1])
            if candidate not in self.graph:
                continue
            on_enemy = self._enemy_home(candidate)
            if on_enemy:
                # Border collision is safe once the tile is on our side.
                minimum = min(self._maze(candidate, ghost) for ghost, _ in active)
                after_enemy = min((min(self._maze(candidate, reply) for reply in
                                       [ghost] + self.graph.get(ghost, []))
                                   for ghost, _ in active), default=99)
                risk = (90 if minimum == 0 else 27 if after_enemy == 0 else
                        7 if after_enemy == 1 else 1 if minimum <= 3 else 0)
            else:
                risk = 0
                after_enemy = 9
            progress = (self._maze(candidate, goal) if goal is not None else 0)
            anti_loop = self.recent.count(candidate)
            score = -risk - .9 * progress - .65 * anti_loop
            if a == Directions.STOP:
                score -= 3
            if a == original:
                score += .3
            if not on_enemy:
                score += 12 + carrying
            options.append((score, a, risk, after_enemy))
        if not options:
            return original
        initial = next((v for v in options if v[1] == original), None)
        best = max(options, key=lambda x: x[0])
        if initial is None:
            return original
        # Override only for meaningful risk difference; don't compromise long
        # horizon planning for a near-tie in the tactical one-step game.
        if initial[2] > best[2] + 7 and best[0] > initial[0] + 2:
            return best[1]
        return original

    # Deprecated experimental hook; kept outside official final selection until
    # controlled tests show it helps in blocked-border situations.
    ENABLE_BREAK_STALE = True

    def choose_action(self, game_state):
        action = super().choose_action(game_state)
        if not self.ENABLE_BREAK_STALE:
            return action
        pos = self._cell(game_state.get_agent_position(self.index))
        if pos is None or self.last_mode != 'attack':
            return action
        carried = game_state.get_agent_state(self.index).num_carrying
        if carried or self.goal_kind == 'return':
            self.breakout_portal = None
            self.breakout_expiry = -1
            return action
        tick = 1200 - getattr(game_state.data, 'timeleft', 1200)
        stale = len(self.recent) >= 9 and len(set(self.recent)) <= 3
        if not hasattr(self, 'breakout_portal'):
            self.breakout_portal = None
            self.breakout_expiry = -1
        # No repeated switching from an anti-loop plan back to normal routing.
        if self.breakout_portal is not None and tick >= self.breakout_expiry:
            self.breakout_portal = None
        if stale and self.breakout_portal is None and self.enemy_portals:
            ghost_positions = [p for i,p,s in self._enemy_threat(game_state,tick)[0]
                               if s.scared_timer<=2]
            mate = self._cell(game_state.get_agent_position(self.other))
            # Prefer an unguarded crossover that can be reached without an
            # unreasonable diversion; also split two attackers geographically.
            choices = []
            for portal in self.enemy_portals:
                near_ghost = min((self._maze(portal, g) for g in ghost_positions), default=7)
                near_mate = self._maze(portal,mate) if mate is not None else 8
                value = (1.7 * min(near_ghost, 7) - .55 * self._maze(pos,portal)
                         + .15 * min(near_mate, 9))
                choices.append((value, portal))
            self.breakout_portal = max(choices)[1]
            self.breakout_expiry = tick + 64
        if self.breakout_portal is not None and self._home(pos):
            _, paths = self._routes(pos, {})
            planned = paths.get(self.breakout_portal)
            if planned in game_state.get_legal_actions(self.index):
                return planned
        return action


class CooperativeTacticsMixin:
    """Optional cooperative tasks using public observations and team messages.

    The host supplies _tick, _team_memory and normal attack/defense planning.
    Return trips are preserved except for a nearby rescue capsule, followed
    immediately by a locked bank trip.
    """

    CAPSULE_WAIT_LIMIT = 10
    CAPSULE_CAMP_LIMIT = 60
    DECOY_LIMIT = 8

    def _reset_cooperative_state(self):
        self.support_wait_turns = 0
        self._support_capsule = None
        self._support_exhausted = set()
        self._support_resume_bank = False
        self._support_return_capsule = None
        self._decoy_turns = 0
        self._decoy_until = -1
        self._decoy_observation = None
        self.strategy_events = {}

    def _strategy_event(self, name):
        if not hasattr(self, 'strategy_events'):
            self.strategy_events = {}
        self.strategy_events[name] = self.strategy_events.get(name, 0) + 1

    def _cooperative_setup(self):
        if not hasattr(self, '_support_exhausted'):
            events = getattr(self, 'strategy_events', {})
            self._reset_cooperative_state()
            self.strategy_events = events
        if not hasattr(self, 'strategy_events'):
            self.strategy_events = {}

    def _support_mate_message(self, state):
        memory = getattr(self, '_team_memory', {})
        message = memory.get(self.other, {})
        # A stale announcement must not create a rescue/decoy task.
        if not isinstance(message, dict):
            return {}
        if self._tick(state) - message.get('tick', -9999) > 8:
            return {}
        return message

    def _cooperative_visible_ghosts(self, state):
        # Last-seen locations are useful for navigation, but are not evidence
        # that a teammate currently needs a capsule or a decoy.
        result = []
        for enemy in self.enemies:
            location = self._cell(state.get_agent_position(enemy))
            status = state.get_agent_state(enemy)
            if (location is not None and not status.is_pacman
                    and status.scared_timer <= 2):
                result.append(location)
        return result

    def _capsule_rescue_needed(self, state, pos, mate, ghosts):
        if not ghosts:
            return False
        for attacker in (pos, mate):
            if attacker is None or self._home(attacker):
                continue
            if min(self._maze(attacker, ghost) for ghost in ghosts) <= 4:
                return True
            # The host can provide branch mouths. A ghost can block the exit
            # before getting close to an attacker near the end of a corridor.
            mouth_map = getattr(self, 'trap_mouth', {})
            mouth = mouth_map.get(attacker)
            if mouth is not None:
                escape = self._maze(attacker, mouth)
                if min(self._maze(ghost, mouth) for ghost in ghosts) <= escape + 1:
                    return True
        return False

    def _support_first_step_safe(self, pos, action, ghosts, capsule=None):
        delta = DIRECTION_STEP.get(action, (0, 0))
        nxt = (pos[0] + delta[0], pos[1] + delta[1])
        if nxt not in self.graph:
            return False
        # Eating the capsule changes collision rules before death is checked.
        if nxt == capsule or self._home(nxt):
            return True
        return all(self._maze(nxt, ghost) >= 2 for ghost in ghosts)

    def _cooperative_attack(self, state, pos, legal, threats, carrying, mate,
                            normal):
        self._cooperative_setup()
        strategy = getattr(self, 'strategy', 'balanced')
        if strategy == 'capsule_wait12':
            strategy = 'capsule_wait'
        capsule_camp = strategy == 'capsule_camp'
        if capsule_camp:
            strategy = 'capsule_wait'
        if strategy == 'decoy':
            self._cooperative_observe_decoy(state, mate)
        if self._support_resume_bank:
            if not carrying or self._home(pos):
                self._support_resume_bank = False
                self._support_return_capsule = None
            else:
                available = [self._cell(c) for c in self.get_capsules(state)]
                capsule = self._support_return_capsule
                risk = self._danger_map(threats, carrying)
                if capsule in available:
                    risk[capsule] = 0
                costs, paths = self._routes(pos, risk)
                ghosts = self._cooperative_visible_ghosts(state)
                if capsule in available and capsule in paths:
                    action = paths[capsule]
                    if action in legal and self._support_first_step_safe(pos, action, ghosts, capsule):
                        return action, capsule, 'capsule-rescue'
                portals = [p for p in self.home_portals if p in paths]
                if portals:
                    goal = min(portals, key=lambda p: costs[p])
                    return paths[goal], goal, 'return'
        if strategy in ('capsule_wait', 'capsule_rescue'):
            rescued = self._cooperative_capsule_rescue(
                state, pos, legal, threats, carrying, mate, normal,
                2 if strategy == 'capsule_wait' else 4)
            if rescued is not None:
                return rescued
        if normal[2] == 'return':
            self.support_wait_turns = 0
            return normal
        if strategy == 'decoy':
            return self._cooperative_decoy(state, pos, legal, carrying, mate, normal)
        if strategy == 'capsule_rescue':
            return normal
        capsule_strategy = strategy in ('capsule_rush', 'capsule_wait')
        # The balanced policy may opt into the measured simple capsule rush.
        if strategy in ('balanced', 'bank3', 'bank5', 'bank8', 'bank12',
                        'secure_lead', 'secure_late', 'harvest_all'):
            capsule_strategy = getattr(self, 'BALANCED_CAPSULE_RUSH', True)
        if not capsule_strategy or self.primary:
            return normal
        capsules = [self._cell(c) for c in self.get_capsules(state)]
        if not capsules or len(self.get_food(state).as_list()) <= 2:
            self.support_wait_turns = 0
            self._support_capsule = None
            return normal
        if carrying >= 3:
            return normal
        ghosts = self._cooperative_visible_ghosts(state)
        risk = self._danger_map(threats, carrying)
        costs, paths = self._routes(pos, risk)
        if strategy == 'capsule_wait' and normal[2] == 'capsule':
            # Native V3 may already select the capsule. A standby experiment
            # must actually defer that consumption when no rescue is needed.
            food_risk = dict(risk)
            for available in capsules:
                food_risk[available] = food_risk.get(available, 0) + 10000
            food_costs, food_paths = self._routes(pos, food_risk)
            food, _ = self._best_food(state, pos, food_costs,
                                      self.get_food(state).as_list(), carrying,
                                      mate, ghosts)
            if food is not None and food in food_paths:
                replacement = food_paths[food]
                if replacement in legal:
                    normal = replacement, food, 'food'
        choices = [c for c in capsules if c in paths]
        if not choices:
            return normal
        capsule = min(choices, key=lambda c: costs.get(c, 9999))
        remaining = getattr(state.data, 'timeleft', 1200)
        # Leave enough actual turns for the mission and a useful food trip.
        if remaining < 4 * (self._maze(pos, capsule) + 12):
            return normal
        if strategy != 'capsule_wait':
            action = paths[capsule]
            if action in legal and self._support_first_step_safe(pos, action, ghosts, capsule):
                if self._support_capsule != capsule:
                    self._strategy_event('capsule_rush_started')
                self._support_capsule = capsule
                return action, capsule, 'capsule-rush'
            return normal

        # Standby/rescue is deliberately a temporary assignment. A teammate at
        # home cannot exploit the safety window yet; collecting is better then.
        if mate is None or self._home(mate):
            self.support_wait_turns = 0
            return normal
        rescue = self._capsule_rescue_needed(state, pos, mate, ghosts)
        if rescue:
            action = paths[capsule]
            if action in legal and self._support_first_step_safe(pos, action, ghosts, capsule):
                self._strategy_event('capsule_rescue_turns')
                self.support_wait_turns = 0
                return action, capsule, 'capsule-rescue'
            return normal
        if capsule in self._support_exhausted:
            return normal
        if self._support_capsule != capsule:
            self._support_capsule = capsule
            self.support_wait_turns = 0
            self._strategy_event('capsule_standby_started')
        # Routing to an adjacent square must not accidentally eat the capsule.
        staging_risk = dict(risk)
        staging_risk[capsule] = staging_risk.get(capsule, 0) + 10000
        stage_costs, stage_paths = self._routes(pos, staging_risk)
        stages = [p for p in self.graph.get(capsule, ())
                  if self._enemy_home(p) and p in stage_paths
                  and all(self._maze(p, ghost) >= 3 for ghost in ghosts)]
        if not stages:
            return normal
        staging = min(stages, key=lambda p: stage_costs.get(p, 99999))
        if stage_costs[staging] >= 9999:
            return normal
        if pos == staging:
            self.support_wait_turns += 1
            self._strategy_event('capsule_standby_turns')
            wait_limit = self.CAPSULE_CAMP_LIMIT if capsule_camp else self.CAPSULE_WAIT_LIMIT
            if self.support_wait_turns >= wait_limit:
                self._support_exhausted.add(capsule)
                self._strategy_event('capsule_standby_timeout')
                return normal
            if Directions.STOP in legal:
                return Directions.STOP, staging, 'capsule-standby'
            return normal
        action = stage_paths[staging]
        if action in legal and self._support_first_step_safe(pos, action, ghosts):
            return action, staging, 'capsule-stage'
        return normal

    def _cooperative_capsule_rescue(self, state, pos, legal, threats, carrying,
                                    mate, normal, max_distance):
        ghosts = self._cooperative_visible_ghosts(state)
        if not self._capsule_rescue_needed(state, pos, mate, ghosts):
            return None
        capsules = [self._cell(c) for c in self.get_capsules(state)]
        nearby = [c for c in capsules if self._maze(pos, c) <= max_distance]
        if not nearby:
            return None
        risk = self._danger_map(threats, carrying)
        for capsule in nearby:
            # A ghost on the destination cannot kill us after consumption.
            risk[capsule] = 0
        costs, paths = self._routes(pos, risk)
        bank_cost = min((costs.get(portal, 9999) for portal in self.home_portals),
                        default=9999)
        choices = [c for c in nearby if c in paths]
        if not choices:
            return None
        capsule = min(choices, key=lambda c: costs.get(c, 9999))
        if normal[2] == 'return' and costs[capsule] > bank_cost + 2:
            return None
        action = paths[capsule]
        if action not in legal or not self._support_first_step_safe(pos, action, ghosts, capsule):
            return None
        self._strategy_event('capsule_rescue_turns')
        if normal[2] == 'return':
            self._strategy_event('capsule_return_rescue')
            self._support_resume_bank = True
            self._support_return_capsule = capsule
        self.support_wait_turns = 0
        return action, capsule, 'capsule-rescue'

    def _cooperative_decoy(self, state, pos, legal, carrying, mate, normal):
        tick = self._tick(state)
        if self.primary or carrying or self._home(pos) or mate is None or self._home(mate):
            self._decoy_turns = 0
            return normal
        message = self._support_mate_message(state)
        mate_carrying = state.get_agent_state(self.other).num_carrying
        # Public teammate state is exact. Messages are only intentions.
        if mate_carrying < 1:
            self._decoy_turns = 0
            return normal
        if tick < self._decoy_until:
            return normal
        ghosts = self._cooperative_visible_ghosts(state)
        candidates = [g for g in ghosts if 2 <= self._maze(pos, g) <= 7
                      and self._maze(mate, g) <= 9
                      and self._maze(pos, g) <= self._maze(mate, g) + 2]
        if not candidates:
            self._decoy_turns = 0
            return normal
        if self._decoy_turns >= self.DECOY_LIMIT:
            self._decoy_turns = 0
            self._decoy_until = tick + 64
            self._strategy_event('decoy_timeout')
            return normal
        ghost = min(candidates, key=lambda g: self._maze(mate, g))
        exits = list(self.home_portals) + [self._cell(c) for c in self.get_capsules(state)]
        escape_now = min((self._maze(pos, e) for e in exits), default=999)
        mate_distance = self._maze(pos, mate)
        options = []
        for action in legal:
            if action == Directions.STOP:
                continue
            dx, dy = DIRECTION_STEP.get(action, (0, 0))
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt not in self.graph or not self._enemy_home(nxt):
                continue
            distance = self._maze(nxt, ghost)
            if not 3 <= distance <= 5:
                continue
            if distance >= self._maze(mate, ghost):
                continue
            if not self._support_first_step_safe(pos, action, ghosts):
                continue
            # Do not draw the defender toward the loaded teammate or deeper
            # into a dead end without a viable route to safety.
            separation = self._maze(nxt, mate)
            escape = min((self._maze(nxt, e) for e in exits), default=999)
            if separation < mate_distance or escape > escape_now:
                continue
            mouth = getattr(self, 'trap_mouth', {}).get(nxt)
            if mouth is not None and self._maze(ghost, mouth) <= self._maze(nxt, mouth) + 1:
                continue
            planned = message.get('goal')
            future_separation = self._maze(nxt, planned) if planned in self.graph else separation
            score = separation + .25 * future_separation - .6 * escape - abs(distance - 4)
            score -= .7 * self.recent.count(nxt)
            options.append((score, action, nxt))
        if not options:
            return normal
        _, action, goal = max(options)
        if not self._decoy_turns:
            self._strategy_event('decoy_started')
        self._decoy_turns += 1
        self._strategy_event('decoy_turns')
        for enemy in self.enemies:
            if self._cell(state.get_agent_position(enemy)) == ghost:
                self._decoy_observation = {'enemy': enemy, 'ghost': ghost,
                                           'decoy': goal, 'tick': tick}
                break
        return action, goal, 'decoy'

    def _cooperative_observe_decoy(self, state, mate):
        previous = self._decoy_observation
        self._decoy_observation = None
        if not previous or mate is None or self._home(mate):
            return
        if self._tick(state) - previous['tick'] > 8:
            return
        enemy = previous['enemy']
        ghost = self._cell(state.get_agent_position(enemy))
        if ghost is None or state.get_agent_state(enemy).is_pacman:
            return
        old = previous['ghost']
        target = previous['decoy']
        # This records visible geometry, not an assumption about the enemy's
        # intent: its move approached our previous decoy tile and moved farther
        # from the loaded teammate's current tile.
        if (self._maze(ghost, target) < self._maze(old, target)
                and self._maze(ghost, mate) > self._maze(old, mate)):
            self._strategy_event('decoy_chase_observed')

    def _cooperative_defense_goal(self, state, pos, invaders, tick, normal):
        self._cooperative_setup()
        strategy = getattr(self, 'strategy', 'balanced')
        if strategy not in ('guard_capsule', 'guard_permanent'):
            return normal
        if state.get_agent_state(self.index).scared_timer > 0:
            return normal
        capsules = [self._cell(c) for c in self.get_capsules_you_are_defending(state)]
        if not capsules:
            return normal
        options = []
        for capsule in capsules:
            for _, intruder, _ in invaders:
                enemy_eta = self._maze(intruder, capsule)
                if enemy_eta > 7:
                    continue
                for gate in self.graph.get(capsule, ()):
                    if not self._home(gate):
                        continue
                    # Gate must be on a shortest capsule approach, and we
                    # must occupy it strictly before the invader can arrive.
                    enemy_gate = self._maze(intruder, gate)
                    our_gate = self._maze(pos, gate)
                    if enemy_gate + 1 == enemy_eta and our_gate + 1 < enemy_gate:
                        options.append((our_gate - enemy_eta, gate))
        if options:
            self._strategy_event('capsule_intercept_turns')
            return min(options)[1], 'capsule-intercept'
        if strategy == 'guard_permanent':
            urgent = any(self._maze(pos, intruder) <= 4
                         or min(self._maze(intruder, capsule) for capsule in capsules) <= 4
                         for _, intruder, _ in invaders)
            if urgent:
                return normal
        elif normal[1] != 'patrol' or invaders:
            return normal
        # Occupying the capsule itself loses to the consume-before-collision
        # ordering. Cover an adjacent approach instead.
        gates = [p for c in capsules for p in self.graph.get(c, ())
                 if self._home(p)]
        if gates:
            goal = min(gates, key=lambda p: (min(self._maze(p, q) for q in self.home_portals)
                                           if self.home_portals else self._maze(pos, p),
                                           self._maze(pos, p)))
            self._strategy_event('capsule_guard_turns')
            return goal, 'capsule-guard'
        return normal


# This class is assembled into the single-file submission by the development
# workflow. It deliberately uses only observations provided by the engine.
class CoordinatedAgent(CooperativeTacticsMixin, TacticalAgent):
    """Map-aware cooperative agent; experimental policies are explicit options."""

    ENABLE_BREAK_STALE = False
    DEFAULT_STRATEGY = 'capsule_wait'
    STRATEGIES = ('balanced', 'corrected', 'capsule_rush', 'capsule_wait',
                  'capsule_rescue', 'guard_capsule', 'decoy', 'bank3', 'bank5',
                  'bank8', 'bank12', 'harvest_all', 'secure_lead', 'secure_late',
                  'capsule_camp', 'guard_permanent', 'capsule_wait12')

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.initial_timeleft = getattr(game_state.data, 'timeleft', 1200)
        self.initial_food_count = len(self.get_food(game_state).as_list())
        self.returned_to_win = max(1, self.initial_food_count - 2)
        self.strategy_events = {}
        self.strategy = getattr(self, 'strategy', self.DEFAULT_STRATEGY)
        self.__dict__.pop('_support_exhausted', None)
        self._cooperative_setup()
        # A factory-scoped dictionary is shared by our two agents only. Reset
        # it on setup so repeated games never reuse targets or a defender.
        if not hasattr(self, '_team_memory'):
            self._team_memory = {}
        self._team_memory.clear()
        self._last_action_pos = None
        self._return_portal = None
        self._return_replan_tick = -999

    def _tick(self, state):
        return self.initial_timeleft - getattr(state.data, 'timeleft', self.initial_timeleft)

    def _event(self, name):
        self.strategy_events[name] = self.strategy_events.get(name, 0) + 1

    def _dead_end_depth(self):
        """Distance OUT of a trap, including single-entrance cyclic rooms.

        Articulation gates separate regions from all border crossings. A
        multi-source BFS establishes which side of a gate is the trapped side.
        No recursive graph walk or map-specific coordinates are needed.
        """
        graph = self.graph
        degree = {p: len(adj) for p, adj in graph.items()}
        queue = deque(p for p in graph if degree[p] <= 1)
        removed = set()
        while queue:
            p = queue.popleft()
            if p in removed:
                continue
            removed.add(p)
            for n in graph[p]:
                if n not in removed:
                    degree[n] -= 1
                    if degree[n] == 1:
                        queue.append(n)
        core = set(graph) - removed
        anchors = set(self.home_portals + self.enemy_portals)
        roots = core or anchors or {self.start}
        depths = {p: 0 for p in roots}
        mouth = {}
        queue = deque(sorted(roots))
        while queue:
            p = queue.popleft()
            for n in graph[p]:
                if n not in depths:
                    depths[n] = depths[p] + 1
                    mouth[n] = p if p in roots else mouth.get(p, p)
                    queue.append(n)

        # Iterative Tarjan traversal finds articulation points even on maps
        # larger than Python's recursion limit.
        discovered, low, parent, children, gates = {}, {}, {}, {}, set()
        serial = 0
        for root in graph:
            if root in discovered:
                continue
            serial += 1
            discovered[root] = low[root] = serial
            parent[root] = None
            children[root] = 0
            stack = [(root, iter(graph[root]))]
            while stack:
                p, neighbors = stack[-1]
                n = next(neighbors, None)
                if n is None:
                    stack.pop()
                    par = parent[p]
                    if par is None:
                        if children[p] > 1:
                            gates.add(p)
                    else:
                        low[par] = min(low[par], low[p])
                        if parent[par] is not None and low[p] >= discovered[par]:
                            gates.add(par)
                    continue
                if n not in discovered:
                    parent[n] = p
                    children[p] += 1
                    children[n] = 0
                    serial += 1
                    discovered[n] = low[n] = serial
                    stack.append((n, iter(graph[n])))
                elif n != parent[p]:
                    low[p] = min(low[p], discovered[n])
        self.articulation_points = gates
        # Leaf pruning already handles ordinary branches. Gate analysis adds
        # rooms with a loop inside but only a single way back to the border.
        if anchors:
            for gate in sorted(gates):
                reached = set(anchors) - {gate}
                queue = deque(reached)
                while queue:
                    p = queue.popleft()
                    for n in graph[p]:
                        if n != gate and n not in reached:
                            reached.add(n)
                            queue.append(n)
                trapped = set(graph) - reached - {gate}
                if not trapped or not (trapped & core):
                    continue
                gate_dist = {gate: 0}
                queue = deque([gate])
                while queue:
                    p = queue.popleft()
                    for n in graph[p]:
                        if n in trapped and n not in gate_dist:
                            gate_dist[n] = gate_dist[p] + 1
                            queue.append(n)
                for p in trapped:
                    if gate_dist.get(p, 0) > depths.get(p, 0):
                        depths[p] = gate_dist[p]
                        mouth[p] = gate
        self.trap_mouth = mouth
        return {p: d for p, d in depths.items() if d > 0}

    def _safe_food(self, state, pos, target, threats):
        gate = self.trap_mouth.get(target)
        if gate is None or self._home(target):
            return True
        escape_eta = self._maze(pos, target) + self._maze(target, gate) + 1
        for enemy, location, status in threats:
            # Last-seen positions are useful soft costs, but are not hard
            # proof that a moving enemy can seal a particular branch.
            if state.get_agent_position(enemy) is None:
                continue
            earliest_danger = max(self._maze(location, gate), status.scared_timer)
            if earliest_danger <= escape_eta + 1:
                return False
        return True

    def _bank_target(self, state):
        amounts = {'bank3': 3, 'bank5': 5, 'bank8': 8, 'bank12': 12,
                   'capsule_wait12': 12,
                   'harvest_all': 100, 'secure_lead': 5, 'secure_late': 5}
        return amounts.get(self.strategy, 8)

    def _holding_lead(self, state):
        score = self.get_score(state)
        left = getattr(state.data, 'timeleft', 1200)
        if self.strategy == 'secure_lead':
            return score >= 5
        if self.strategy == 'secure_late':
            return score >= 3 and left <= 320
        return False

    def _assigned_defender(self, state, pos, invaders, tick):
        invading = [i for i in self.enemies if state.get_agent_state(i).is_pacman]
        if not invading:
            if self.get_score(state) > 0 and getattr(state.data, 'timeleft', 1200) < 200:
                return max(self.team)
            return None
        targets = [p for _, p, _ in invaders]
        if not targets and self.incident is not None and tick - self.incident_t < 28:
            targets = [self.incident]
        if not targets:
            for enemy in invading:
                possible = self.beliefs.get(enemy, {})
                if possible:
                    targets.append(max(possible, key=possible.get))
        if not targets:
            targets = self.home_portals or [self.start]
        costs = {}
        for teammate in self.team:
            location = self._cell(state.get_agent_position(teammate))
            own = state.get_agent_state(teammate)
            # Loaded attackers commit to banking in _should_defend. They
            # cannot also be promised as the teammate's active defender.
            if own.is_pacman and own.num_carrying:
                continue
            distance = min(self._maze(location, target) for target in targets)
            # Cargo is banked before a distant attacker abandons its trip.
            cargo_cost = 4 * own.num_carrying if own.is_pacman else 0
            home_bonus = -2 if not own.is_pacman else 0
            scared_cost = min(18, own.scared_timer)
            costs[teammate] = distance + cargo_cost + scared_cost + home_bonus
            if teammate == min(self.team):
                costs[teammate] += 1.5
        if not costs:
            return None
        chosen = min(costs, key=lambda i: (costs[i], i))
        previous = self._team_memory.get('_defender')
        if previous and tick - previous['tick'] < 12:
            old = previous['index']
            if old in costs and costs[old] <= costs[chosen] + 3:
                chosen = old
        self._team_memory['_defender'] = {'index': chosen, 'tick': tick}
        return chosen

    def _should_defend(self, state, pos, invaders, tick):
        own = state.get_agent_state(self.index)
        if own.is_pacman and own.num_carrying:
            return False
        if self.strategy == 'guard_permanent' and not self.primary:
            if self.get_capsules_you_are_defending(state):
                return True
        if not self.primary and self.strategy == 'capsule_camp' and self.get_capsules(state):
            # This deliberately stronger camping experiment includes the
            # opportunity cost of leaving our half without its usual guard.
            return False
        if not self.primary and self.strategy == 'decoy':
            mate = state.get_agent_state(self.other)
            if mate.is_pacman and mate.num_carrying >= 2:
                return False
        # These explicit experiments stop both agents attacking after a lead.
        # This is a heuristic policy, never a guarantee against enemy cargo.
        if self._holding_lead(state) and not own.is_pacman:
            return True
        defender = self._assigned_defender(state, pos, invaders, tick)
        if defender is not None:
            return self.index == defender
        return False

    def _best_food(self, state, pos, costs, food, carrying, mate, live_threats):
        threats, _ = self._enemy_threat(state, self._tick(state))
        candidates = []
        mate_message = self._team_memory.get(self.other, {})
        mate_busy = (mate_message.get('mode') == 'defend' or
                     mate_message.get('kind') in ('capsule', 'capsule-stage', 'capsule-rush',
                                                  'capsule-rescue', 'capsule-standby', 'decoy', 'return'))
        mate_available = mate is not None and not mate_busy
        if mate is not None:
            mate_available = mate_available and state.get_agent_state(self.other).num_carrying < self._bank_target(state)
        for target in food:
            if target not in costs or not self._safe_food(state, pos, target, threats):
                continue
            cluster = min(5, sum(self._maze(target, q) <= 3 for q in food))
            value = 13 - costs[target] + 2.25 * cluster - .13 * self._bank_distance(target)
            if self.goal_kind == 'food' and target == self.goal:
                value += 5
            value -= self.recent.count(target) * 1.5
            branch = self.trap_mouth.get(target, target)
            reserved = mate_message.get('goal')
            if (reserved is not None and mate_message.get('kind') == 'food' and
                    self._tick(state) - mate_message.get('tick', -999) <= 8 and
                    self.trap_mouth.get(reserved, reserved) == branch):
                value -= 8
            mate_value = (13 - self._maze(mate, target) + 2.25 * cluster -
                          .13 * self._bank_distance(target)) if mate_available else 0
            candidates.append((value, target, branch, mate_value))
        if not candidates:
            return None, -1e10
        if mate_available and len(candidates) > 1:
            # Exhaustive two-task assignment is cheap (at most food_count^2)
            # and uses both travel costs, unlike a one-sided crowding penalty.
            best_pair = None
            for value, target, branch, _ in candidates:
                mate_best = max(other[3] - (12 if other[2] == branch else 0)
                                for other in candidates if other[1] != target)
                pair = (value + .65 * mate_best, value, target)
                if best_pair is None or pair > best_pair:
                    best_pair = pair
            return best_pair[2], best_pair[1]
        winner = max(candidates, key=lambda item: (item[0], item[1]))
        return winner[1], winner[0]

    def _return_route(self, pos, threats, carrying):
        costs, first = self._routes(pos, self._danger_map(threats, carrying))
        portals = [p for p in self.home_portals if p in costs]
        if portals:
            target = min(portals, key=lambda p: (costs[p], p))
            previous = self._return_portal
            if previous in costs and previous in portals and costs[previous] <= costs[target] + 16:
                target = previous
            tick = getattr(self, '_current_tick', 0)
            stale = len(self.recent) >= 9 and len(set(self.recent)) <= 3
            if stale and tick - self._return_replan_tick >= 24:
                alternatives = [p for p in portals if previous is None or self._maze(p, previous) >= 4]
                if alternatives:
                    target = min(alternatives, key=lambda p: (costs[p], p))
                    self._return_replan_tick = tick
                    self._event('return_breakout')
            self._return_portal = target
            return first[target], target, 'return'
        return Directions.STOP, pos, 'return'

    def _choose_attack(self, state, pos, legal, threats, carrying, mate):
        food = self.get_food(state).as_list()
        active = [p for _, p, status in threats if status.scared_timer <= 2]
        nearest = min((self._maze(pos, p) for p in active), default=999)
        remaining_turns = max(0, getattr(state.data, 'timeleft', 1200) // 4)
        bank_distance = self._bank_distance(pos)
        score = self.get_score(state)
        returned = sum(state.get_agent_state(i).num_returned for i in self.team)
        should_return = carrying > 0 and (
            self.goal_kind == 'return' or carrying >= self._bank_target(state) or
            returned + carrying >= self.returned_to_win or
            remaining_turns <= bank_distance + 5 or len(food) <= 2 or
            (carrying >= 3 and nearest <= 5) or nearest <= 3 or
            (self.strategy == 'secure_lead' and score + carrying >= 5) or
            (self.strategy == 'secure_late' and remaining_turns <= 80 and score + carrying >= 5) or
            (remaining_turns < 30 and score <= 0 < score + carrying))
        risk = self._danger_map(threats, carrying)
        costs, paths = self._routes(pos, risk)
        if should_return:
            normal = self._return_route(pos, threats, carrying)
        else:
            target, value = self._best_food(state, pos, costs, food, carrying, mate, active)
            kind = 'food'
            for capsule in self.get_capsules(state):
                capsule_value = 9 - costs.get(capsule, 9999)
                if active:
                    capsule_value += 10 if nearest <= 7 else 4
                if capsule_value > value:
                    target, value, kind = capsule, capsule_value, 'capsule'
            if target is None:
                if carrying:
                    normal = self._return_route(pos, threats, carrying)
                else:
                    portals = self.enemy_portals or [self.start]
                    target = min(portals, key=lambda p: costs.get(p, 9999) +
                                 3 * self.recent.count(p))
                    normal = paths.get(target, Directions.STOP), target, 'probe'
            else:
                normal = paths.get(target, Directions.STOP), target, kind
        return self._cooperative_attack(state, pos, legal, threats, carrying, mate, normal)

    def _defense_goal(self, state, pos, invaders, tick):
        if invaders and state.get_agent_state(self.index).scared_timer == 0:
            enemy_id, enemy, status = min(invaders, key=lambda item:
                                          self._maze(pos, item[1]) - 1.2 * item[2].num_carrying)
            gate = self.trap_mouth.get(enemy)
            if gate is not None and self._home(gate):
                if self._maze(pos, gate) + 1 < self._maze(enemy, gate):
                    return self._cooperative_defense_goal(
                        state, pos, invaders, tick, (gate, 'branch-intercept'))
            if status.num_carrying >= 3 and self.opponent_mode.get(enemy_id) == 'retreat':
                exits = sorted(self.home_portals, key=lambda p: self._maze(enemy, p))[:2]
                for gate in exits:
                    eta = self._maze(enemy, gate)
                    if eta <= 8 and self._maze(pos, gate) + 1 < eta:
                        return self._cooperative_defense_goal(
                            state, pos, invaders, tick, (gate, 'exit-intercept'))
        normal = super()._defense_goal(state, pos, invaders, tick)
        if not invaders and self._holding_lead(state) and self.home_portals:
            other = self._team_memory.get(self.other, {}).get('goal')
            choices = [p for p in self.home_portals if other is None or self._maze(p, other) >= 4]
            if choices:
                normal = min(choices, key=lambda p: self._maze(pos, p)), 'lead-patrol'
        return self._cooperative_defense_goal(state, pos, invaders, tick, normal)

    def _choose_defense(self, state, pos, legal, threats, invaders, tick):
        own = state.get_agent_state(self.index)
        if own.is_pacman and own.num_carrying:
            return self._return_route(pos, threats, own.num_carrying)
        return super()._choose_defense(state, pos, legal, threats, invaders, tick)

    def _tactical_action(self, state, pos, legal, threats, carrying, normal):
        """Conservative one-round reply check around visible active ghosts.

        A ghost can move once before our next decision. Staying one square
        from it abroad is therefore an immediate losing action. Capsule
        consumption happens before collision and is explicitly exempted.
        """
        active = [p for enemy, p, status in threats if status.scared_timer <= 1
                  and state.get_agent_position(enemy) is not None]
        if not active or min(self._maze(pos, p) for p in active) > 3:
            return normal
        capsules = set(self.get_capsules(state))
        original, goal, kind = normal
        choices = []
        for action in legal:
            dx, dy = DIRECTION_STEP[action]
            nxt = pos[0] + dx, pos[1] + dy
            distance = min(self._maze(nxt, enemy) for enemy in active)
            unsafe = self._enemy_home(nxt) and nxt not in capsules and distance <= 1
            value = (-1000 if unsafe else 0) - .8 * self._maze(nxt, goal)
            value -= .5 * self.recent.count(nxt)
            if action == original:
                value += 1
            if self._home(nxt):
                value += 8 + carrying
            if action == Directions.STOP:
                value -= .2
            choices.append((value, action, unsafe))
        original_choice = next((c for c in choices if c[1] == original), None)
        best = max(choices)
        if original_choice and original_choice[2] and not best[2]:
            self._event('tactical_escape')
            return best[1], goal, kind
        return normal

    def choose_action(self, state):
        # Core choice dispatches through this class's role/attack/defense hooks.
        # Its elapsed counter is replaced here to support arbitrary game lengths.
        tick = self._tick(state)
        self._current_tick = tick
        legal = state.get_legal_actions(self.index)
        pos = self._cell(state.get_agent_position(self.index))
        if not legal or pos is None:
            return Directions.STOP
        if pos == self.start and self._last_action_pos is not None and self._maze(pos, self._last_action_pos) > 1:
            self.goal = self.goal_kind = None
            self.recent.clear()
        self._last_action_pos = pos
        self.recent.append(pos)
        self._update_food_incident(state, tick)
        self._update_beliefs(state, pos, tick)
        threats, invaders = self._enemy_threat(state, tick)
        mate = self._cell(state.get_agent_position(self.other))
        carrying = state.get_agent_state(self.index).num_carrying
        if not carrying or self._home(pos):
            self._return_portal = None
        if self._should_defend(state, pos, invaders, tick):
            normal = self._choose_defense(state, pos, legal, threats, invaders, tick)
            self.last_mode = 'defend'
        else:
            normal = self._choose_attack(state, pos, legal, threats, carrying, mate)
            normal = self._tactical_action(state, pos, legal, threats, carrying, normal)
            self.last_mode = 'attack'
        action, self.goal, self.goal_kind = normal
        if action not in legal:
            self._event('illegal_action_fallback')
            action = Directions.STOP if Directions.STOP in legal else legal[0]
        self.last_turn = tick
        self._team_memory[self.index] = {
            'tick': tick, 'pos': pos, 'goal': self.goal, 'kind': self.goal_kind,
            'carrying': carrying, 'mode': self.last_mode}
        return action

# Experimental native-observation counter-strategies, assembled onto frozen V4.
_FrozenV4Agent = CoordinatedAgent


class CoordinatedAgent(_FrozenV4Agent):
    COUNTER_FEATURES = ()

    def register_initial_state(self, state):
        super().register_initial_state(state)
        self._counter_previous = {}

    def _update_profiles(self, state, pos, tick):
        self._counter_previous = {
            enemy: self.enemy_profiles[enemy]['last_position'] for enemy in self.enemies}
        return super()._update_profiles(state, pos, tick)

    def _should_defend(self, state, pos, invaders, tick):
        if 'scared_attack' in self.COUNTER_FEATURES:
            own = state.get_agent_state(self.index)
            if own.scared_timer > max(6, self._bank_distance(pos) + 2):
                self._event('scared_counterattack_turns')
                return False
        return super()._should_defend(state, pos, invaders, tick)

    def _bank_target(self, state):
        normal = super()._bank_target(state)
        if 'bold' not in self.COUNTER_FEATURES:
            return normal
        pos = self._cell(state.get_agent_position(self.index))
        active = [self._cell(state.get_agent_position(i)) for i in self.enemies
                  if not state.get_agent_state(i).is_pacman
                  and state.get_agent_state(i).scared_timer <= 2
                  and state.get_agent_position(i) is not None]
        if not active or min(self._maze(pos, p) for p in active) > 7:
            return max(normal, 16)
        return normal

    def _choose_attack(self, state, pos, legal, threats, carrying, mate):
        if 'bold' in self.COUNTER_FEATURES:
            # Last-seen positions remain soft route costs, not current melee facts.
            threats = [t for t in threats if state.get_agent_position(t[0]) is not None]
        normal = super()._choose_attack(state, pos, legal, threats, carrying, mate)
        if 'scared_attack' not in self.COUNTER_FEATURES:
            return normal
        own = state.get_agent_state(self.index)
        if own.scared_timer <= 0 or not self._home(pos):
            return normal
        intruders = [self._cell(state.get_agent_position(i)) for i in self.enemies
                     if state.get_agent_state(i).is_pacman
                     and state.get_agent_position(i) is not None]
        if not intruders or min(self._maze(pos, p) for p in intruders) > 3:
            return normal
        action, goal, kind = normal
        options = []
        for candidate in legal:
            dx, dy = DIRECTION_STEP[candidate]
            nxt = (pos[0] + dx, pos[1] + dy)
            clearance = min(self._maze(nxt, p) for p in intruders)
            danger = self._home(nxt) and clearance <= 1
            value = (-1000 if danger else 0) - self._maze(nxt, goal)
            if candidate == action:
                value += .3
            options.append((value, candidate))
        return max(options)[1], goal, kind

    def _defense_goal(self, state, pos, invaders, tick):
        if 'intercept' not in self.COUNTER_FEATURES or not invaders:
            return super()._defense_goal(state, pos, invaders, tick)
        if state.get_agent_state(self.index).scared_timer:
            return super()._defense_goal(state, pos, invaders, tick)
        # Take an available capture before committing to a distant prediction.
        adjacent = [p for _, p, _ in invaders if self._maze(pos, p) <= 1]
        if adjacent:
            self._event('immediate_capture_turns')
            return min(adjacent, key=lambda p: self._maze(pos, p)), 'capture-now'
        capsules = list(self.get_capsules_you_are_defending(state))
        defended = self.get_food_you_are_defending(state).as_list()
        options = []
        for enemy, enemy_pos, status in invaders:
            destinations = []
            close_caps = [p for p in capsules if self._maze(enemy_pos, p) <= 8]
            for p in close_caps:
                destinations.append((p, 12, 'capsule-cutoff'))
            previous = self._counter_previous.get(enemy)
            home_distance = min((self._maze(enemy_pos, p) for p in self.home_portals), default=999)
            old_distance = min((self._maze(previous, p) for p in self.home_portals), default=999) if previous else 999
            retreating = status.num_carrying >= 6 and home_distance < old_distance
            if retreating or (status.num_carrying >= 12 and home_distance <= 9):
                for p in sorted(self.home_portals, key=lambda p: self._maze(enemy_pos, p))[:2]:
                    destinations.append((p, 8 + min(15, status.num_carrying), 'cargo-cutoff'))
            elif not close_caps and defended:
                for p in sorted(defended, key=lambda p: self._maze(enemy_pos, p))[:2]:
                    destinations.append((p, 5, 'food-cutoff'))
            for destination, value, kind in destinations:
                distance = self._maze(enemy_pos, destination)
                for gate in self.legal_cells:
                    if not self._home(gate):
                        continue
                    enemy_eta = self._maze(enemy_pos, gate)
                    if not 1 <= enemy_eta <= 8:
                        continue
                    ours_eta = self._maze(pos, gate)
                    if ours_eta + 1 > enemy_eta:
                        continue
                    if enemy_eta + self._maze(gate, destination) != distance:
                        continue
                    priority = value - .8 * ours_eta - .25 * enemy_eta
                    priority += 1 if len(self.graph[gate]) <= 2 else 0
                    if self.goal == gate:
                        priority += 1.5
                    options.append((priority, gate, kind))
        if options:
            _, goal, kind = max(options)
            self._event(kind + '_turns')
            return goal, kind
        return super()._defense_goal(state, pos, invaders, tick)



CoordinatedAgent.COUNTER_FEATURES = ('scared_attack', 'intercept', 'bold')

# Restrict aggression to public safety evidence and model reactive escape goals.
_UnconditionalCounter = CoordinatedAgent


class CoordinatedAgent(_UnconditionalCounter):
    def _bank_target(self, state):
        normal = _FrozenV4Agent._bank_target(self, state)
        enemy_states = [state.get_agent_state(i) for i in self.enemies]
        pos = self._cell(state.get_agent_position(self.index))
        protected = min(s.scared_timer for s in enemy_states) > self._bank_distance(pos) + 8
        both_raiding = all(s.is_pacman for s in enemy_states)
        if protected or both_raiding:
            return max(normal, 16)
        return normal

    def _defense_goal(self, state, pos, invaders, tick):
        if not invaders or state.get_agent_state(self.index).scared_timer:
            return _FrozenV4Agent._defense_goal(self, state, pos, invaders, tick)
        adjacent = [p for _, p, _ in invaders if self._maze(pos, p) <= 1]
        if adjacent:
            self._event('immediate_capture_turns')
            return min(adjacent, key=lambda p: self._maze(pos, p)), 'capture-now'
        defended = self.get_food_you_are_defending(state).as_list()
        capsules = self.get_capsules_you_are_defending(state)
        active_defenders = [self._cell(state.get_agent_position(i)) for i in self.team
                            if not state.get_agent_state(i).is_pacman
                            and state.get_agent_state(i).scared_timer == 0]
        options = []
        for enemy, enemy_pos, status in invaders:
            home_distance = min((self._maze(enemy_pos, p) for p in self.home_portals), default=999)
            previous = self._counter_previous.get(enemy)
            old_distance = min((self._maze(previous, p) for p in self.home_portals), default=999) if previous else 999
            threatened = min((self._maze(enemy_pos, p) for p in active_defenders), default=999) <= 3
            retreating = (status.num_carrying >= 6 and home_distance < old_distance)
            if threatened and capsules:
                target = min(capsules, key=lambda p: self._maze(enemy_pos, p))
                destinations = [(target, 14, 'reactive-capsule-cutoff')]
            elif threatened or retreating:
                destinations = [(p, 8 + min(15, status.num_carrying), 'reactive-exit-cutoff')
                                for p in sorted(self.home_portals, key=lambda p: self._maze(enemy_pos, p))[:2]]
            else:
                destinations = [(p, 5, 'reactive-food-cutoff')
                                for p in sorted(defended, key=lambda p: self._maze(enemy_pos, p))[:2]]
            for destination, value, kind in destinations:
                distance = self._maze(enemy_pos, destination)
                for gate in self.legal_cells:
                    if not self._home(gate):
                        continue
                    enemy_eta = self._maze(enemy_pos, gate)
                    ours_eta = self._maze(pos, gate)
                    if not 1 <= enemy_eta <= 8 or ours_eta + 1 > enemy_eta:
                        continue
                    if enemy_eta + self._maze(gate, destination) != distance:
                        continue
                    priority = value - .8 * ours_eta - .25 * enemy_eta
                    priority += 1 if len(self.graph[gate]) <= 2 else 0
                    if self.goal == gate:
                        priority += 1.5
                    options.append((priority, gate, kind))
        if options:
            _, goal, kind = max(options)
            self._event(kind + '_turns')
            return goal, kind
        return _FrozenV4Agent._defense_goal(self, state, pos, invaders, tick)

# Shared evidence ledger: public food changes, cyclic turns, and team sightings.
_BeforeCargoIntelligence = CoordinatedAgent


class CoordinatedAgent(_BeforeCargoIntelligence):
    CARGO_MISSIONS = True
    CARGO_REPLACE_BELIEFS = True

    def register_initial_state(self, state):
        super().register_initial_state(state)
        self._intel_initial_food = len(self.get_food_you_are_defending(state).as_list())
        self._intel_routes = {}
        self._intel_plan_cache = None

    def _update_beliefs(self, state, pos, tick):
        super()._update_beliefs(state, pos, tick)
        self._update_cargo_intelligence(state, tick)

    def _update_cargo_intelligence(self, state, tick):
        self._intel_plan_cache = None
        food = set(self.get_food_you_are_defending(state).as_list())
        capsules = set(self.get_capsules_you_are_defending(state))
        ledger = self._team_memory.get('_cargo_intelligence')
        if ledger is None:
            ledger = {'tick': tick, 'observer': self.index, 'food': food,
                      'capsules': capsules, 'tracks': {}}
            self._team_memory['_cargo_intelligence'] = ledger
        elapsed = max(0, tick - ledger['tick'])
        steps = {i: sum((ledger['observer'] + k) % 4 == i for k in range(elapsed))
                 for i in self.enemies}
        missing = ledger['food'] - food
        added = food - ledger['food']
        eaten = missing | (ledger['capsules'] - capsules)
        movers = [i for i in self.enemies if steps[i]]
        eater = movers[0] if len(movers) == 1 and len(missing) <= steps[movers[0]] else None
        exact_food_position = (next(iter(eaten)) if eater is not None and
                               steps[eater] == 1 and len(eaten) == 1 else None)
        returned = sum(state.get_agent_state(i).num_returned for i in self.enemies)
        total = max(0, self._intel_initial_food - len(food) - returned)
        tracks = ledger['tracks']
        own_position = self._cell(state.get_agent_position(self.index))
        mate_position = self._cell(state.get_agent_position(self.other))
        sonar = state.get_agent_distances()

        for enemy in self.enemies:
            status = state.get_agent_state(enemy)
            visible = self._cell(state.get_agent_position(enemy))
            previous = tracks.get(enemy)
            if previous is None:
                previous = {'lower': 0, 'upper': total, 'positions': {},
                            'returned': status.num_returned, 'evidence_tick': -9999,
                            'last_position': None, 'heading': Directions.STOP}
                tracks[enemy] = previous
            if not status.is_pacman:
                if previous['lower']:
                    self._event('cargo_threat_cleared')
                previous.update(lower=0, upper=0, expected=0.0, positions={},
                                returned=status.num_returned, last_position=None)
                continue

            old_position = previous['last_position']
            old_evidence = previous['evidence_tick']
            if status.num_returned != previous['returned']:
                previous['lower'], previous['upper'] = 0, total
            if added:
                # A capture replenishes food. Do not preserve incompatible loads.
                previous['lower'] = 0
                previous['upper'] = total
            elif missing:
                if eater == enemy:
                    previous['lower'] += len(missing)
                    previous['upper'] += len(missing)
                    self._event('cargo_food_attributed')
                elif eater is None and steps[enemy]:
                    previous['upper'] += min(len(missing), steps[enemy])

            distribution = previous['positions']
            heading = DIRECTION_STEP.get(previous['heading'], (0, 0))
            for _ in range(min(steps[enemy], 8)):
                advanced = {}
                for origin, probability in distribution.items():
                    options = [origin] + list(self.graph.get(origin, ()))
                    weights = [1.6 if (p[0] - origin[0], p[1] - origin[1]) == heading else 1.0
                               for p in options]
                    normalizer = sum(weights)
                    for location, weight in zip(options, weights):
                        advanced[location] = advanced.get(location, 0.0) + probability * weight / normalizer
                distribution = advanced

            inferred = exact_food_position if enemy == eater else None
            evidence = visible if visible is not None else inferred
            if evidence is not None:
                distribution = {evidence: 1.0}
                previous['evidence_tick'] = tick
                previous['last_position'] = evidence
                if old_position is not None and tick - old_evidence <= 4 and steps[enemy]:
                    previous['retreating'] = self._bank_distance(evidence) < self._bank_distance(old_position)
                elif old_position is None or tick - old_evidence > 4:
                    previous['retreating'] = False
                if visible is not None:
                    # Never inspect num_carrying for an unseen enemy.
                    previous['lower'] = previous['upper'] = status.num_carrying
                    previous['heading'] = status.get_direction()
                else:
                    self._event('cargo_hidden_food_fix')
                    if old_position is not None and self._maze(old_position, evidence) == 1:
                        delta = evidence[0] - old_position[0], evidence[1] - old_position[1]
                        previous['heading'] = next((a for a, d in DIRECTION_STEP.items() if d == delta), Directions.STOP)
            else:
                reading = sonar[enemy] if sonar is not None else None
                def allowed(location):
                    return (self._home(location) and
                            self._manhattan(own_position, location) > 5 and
                            (mate_position is None or self._manhattan(mate_position, location) > 5) and
                            (reading is None or abs(self._manhattan(own_position, location) - reading) <= 6))
                posterior = {p: weight for p, weight in distribution.items() if allowed(p)}
                if not posterior:
                    posterior = {p: weight for p, weight in self.beliefs.get(enemy, {}).items() if allowed(p)}
                distribution = posterior
                normalizer = sum(distribution.values())
                if normalizer:
                    distribution = {p: weight / normalizer for p, weight in distribution.items()}
            previous['positions'] = distribution
            previous['returned'] = status.num_returned
            # Native defence/search can also benefit from a fresh, shared food fix.
            if distribution and self.CARGO_REPLACE_BELIEFS:
                self.beliefs[enemy] = dict(distribution)

        # Joint inventory, not two independent guesses at the same missing food.
        first, second = self.enemies
        a, b = tracks[first], tracks[second]
        allocations = [(amount, total - amount) for amount in range(total + 1)
                       if a['lower'] <= amount <= a['upper'] and
                       b['lower'] <= total - amount <= b['upper']]
        if not allocations:
            # Visible loads and public non-invader states remain hard evidence.
            for enemy in self.enemies:
                status = state.get_agent_state(enemy)
                if status.is_pacman and state.get_agent_position(enemy) is None:
                    tracks[enemy]['lower'], tracks[enemy]['upper'] = 0, total
            allocations = [(amount, total - amount) for amount in range(total + 1)
                           if a['lower'] <= amount <= a['upper'] and
                           b['lower'] <= total - amount <= b['upper']]
            self._event('cargo_inventory_reconciled')
        if allocations:
            for offset, enemy in enumerate(self.enemies):
                values = [pair[offset] for pair in allocations]
                tracks[enemy].update(lower=min(values), upper=max(values), expected=sum(values) / len(values))
        else:
            # Defensive fallback for a synthetic/inconsistent observation.
            for enemy in self.enemies:
                tracks[enemy]['expected'] = float(tracks[enemy]['lower'])
        ledger.update(tick=tick, observer=self.index, food=food, capsules=capsules,
                      total=total, allocations=allocations)

    def _intel_home_routes(self, origin):
        if origin in self._intel_routes:
            return self._intel_routes[origin]
        distance, count, first = {origin: 0}, {origin: 1}, {origin: Directions.STOP}
        queue = deque([origin])
        while queue:
            position = queue.popleft()
            for neighbor in self.graph.get(position, ()):
                if not self._home(neighbor):
                    continue
                if neighbor not in distance:
                    distance[neighbor] = distance[position] + 1
                    count[neighbor] = 0
                    delta = neighbor[0] - position[0], neighbor[1] - position[1]
                    action = next(a for a, step in DIRECTION_STEP.items() if step == delta)
                    first[neighbor] = action if position == origin else first[position]
                    queue.append(neighbor)
                if distance[neighbor] == distance[position] + 1:
                    count[neighbor] += count[position]
        result = distance, count, first
        self._intel_routes[origin] = result
        return result

    def _cargo_destinations(self, state, origin, track):
        distances, _, _ = self._intel_home_routes(origin)
        exits = sorted(self.home_portals, key=lambda p: distances.get(p, 999))
        if not exits:
            return []
        nearest = distances.get(exits[0], 999)
        exits = [p for p in exits if distances.get(p, 999) <= nearest + 2]
        cargo = track['expected']
        escape_weight = .8 if cargo >= 8 else .6 if cargo >= 4 else .35
        if track.get('retreating'):
            escape_weight = max(.85, escape_weight)
        defenders = [self._cell(state.get_agent_position(i)) for i in self.team
                     if not state.get_agent_state(i).is_pacman and not state.get_agent_state(i).scared_timer]
        threatened = min((self._maze(origin, p) for p in defenders), default=999) <= 4
        if threatened:
            escape_weight = max(.8, escape_weight)
        capsules = [p for p in self.get_capsules_you_are_defending(state)
                    if distances.get(p, 999) <= nearest + 3]
        capsule_weight = .35 if capsules and threatened else .1 if capsules else 0
        food = sorted(self.get_food_you_are_defending(state).as_list(), key=lambda p: distances.get(p, 999))[:2]
        food_weight = 1 - escape_weight
        destinations = []
        for group, weight in ((exits, escape_weight * (1 - capsule_weight)),
                              (capsules, capsule_weight), (food, food_weight * (1 - capsule_weight))):
            if not group:
                continue
            minimum = min(distances.get(p, 999) for p in group)
            weights = [math.exp(-.65 * (distances.get(p, 999) - minimum)) for p in group]
            normalizer = sum(weights)
            destinations.extend((p, weight * w / normalizer) for p, w in zip(group, weights))
        normalizer = sum(w for _, w in destinations)
        return [(p, w / normalizer) for p, w in destinations] if normalizer else []

    def _cargo_mission(self, state, tick):
        if not self.CARGO_MISSIONS:
            return None
        if self._intel_plan_cache is not None and self._intel_plan_cache[0] == tick:
            return self._intel_plan_cache[1]
        ledger = self._team_memory.get('_cargo_intelligence', {})
        capsules = set(self.get_capsules_you_are_defending(state))
        options = []
        for enemy, track in ledger.get('tracks', {}).items():
            cargo = track.get('expected', 0)
            if cargo < 3 or not state.get_agent_state(enemy).is_pacman:
                continue
            # Unretained probability is not renormalized into confidence.
            origins = sorted(track['positions'].items(), key=lambda item: -item[1])[:12]
            scenarios = [(origin, destination, probability * weight)
                         for origin, probability in origins
                         for destination, weight in self._cargo_destinations(state, origin, track)]
            for teammate in self.team:
                own = state.get_agent_state(teammate)
                position = self._cell(state.get_agent_position(teammate))
                if own.is_pacman or own.scared_timer or position is None:
                    continue
                ours, _, first = self._intel_home_routes(position)
                coverage, weighted_time = {}, {}
                for origin, destination, probability in scenarios:
                    enemy_distance, enemy_paths, _ = self._intel_home_routes(origin)
                    to_destination, destination_paths, _ = self._intel_home_routes(destination)
                    full_distance = enemy_distance.get(destination, 999)
                    for gate, ours_eta in ours.items():
                        enemy_eta = enemy_distance.get(gate, 999)
                        if ours_eta > 10 or not 1 <= enemy_eta <= 12 or gate in capsules:
                            continue
                        ours_arrival = 0 if not ours_eta else (teammate - self.index) % 4 + 4 * (ours_eta - 1)
                        enemy_arrival = (enemy - self.index) % 4 + 4 * (enemy_eta - 1)
                        if ours_arrival > enemy_arrival:
                            continue
                        if enemy_eta + to_destination.get(gate, 999) != full_distance:
                            continue
                        if any(enemy_distance.get(c, 999) + self._intel_home_routes(c)[0].get(gate, 999) == enemy_eta
                               for c in capsules):
                            continue
                        path_fraction = min(1.0, enemy_paths[gate] * destination_paths[gate] / enemy_paths[destination])
                        mass = probability * path_fraction
                        coverage[gate] = coverage.get(gate, 0) + mass
                        weighted_time[gate] = weighted_time.get(gate, 0) + mass * enemy_eta
                for gate, probability in coverage.items():
                    if probability < .4:
                        continue
                    distance = ours[gate]
                    value = (cargo + 3) * probability - .32 * distance - .1 * weighted_time[gate]
                    if teammate == min(self.team):
                        value -= .5
                    previous = self._team_memory.get('_cargo_mission', {})
                    if previous.get('agent') == teammate and previous.get('goal') == gate and previous.get('enemy') == enemy:
                        value += .6
                    if value > 1:
                        options.append((value, teammate, enemy, gate, probability, first[gate], cargo))
        result = None
        if options:
            value, teammate, enemy, gate, probability, action, cargo = max(options)
            result = {'agent': teammate, 'enemy': enemy, 'goal': gate, 'action': action,
                      'coverage': probability, 'cargo': cargo, 'value': value, 'tick': tick}
            self._team_memory['_cargo_mission'] = result
        self._intel_plan_cache = tick, result
        return result

    def _assigned_defender(self, state, pos, invaders, tick):
        plan = self._cargo_mission(state, tick)
        if plan:
            self._event('cargo_defender_assigned')
            return plan['agent']
        return super()._assigned_defender(state, pos, invaders, tick)

    def _choose_defense(self, state, pos, legal, threats, invaders, tick):
        plan = self._cargo_mission(state, tick)
        if plan and plan['agent'] == self.index:
            # Take an immediate capture if it does not delay the carrier blockade.
            adjacent = [(status.num_carrying, location) for _, location, status in invaders
                        if self._maze(pos, location) <= 1 and location not in self.get_capsules_you_are_defending(state)
                        and (status.num_carrying >= plan['cargo'] or
                             self._maze(location, plan['goal']) <= self._maze(pos, plan['goal']))]
            goal = max(adjacent)[1] if adjacent else plan['goal']
            _, _, paths = self._intel_home_routes(pos)
            action = paths.get(goal)
            if action in legal:
                self._event('cargo_intercept_turns')
                return action, goal, 'cargo-intercept'
        return super()._choose_defense(state, pos, legal, threats, invaders, tick)

# Solve the one-defender pursuit game with exits and capsules as escape states.
# A winning entry means capture even against an evader choosing its best reply.
_WithCargoEvidence = CoordinatedAgent


class CoordinatedAgent(_WithCargoEvidence):
    CARGO_REPLACE_BELIEFS = False
    CARGO_MIN_LOAD = 5

    def _cargo_capture_table(self, state):
        capsules = frozenset(self.get_capsules_you_are_defending(state))
        cache = self._team_memory.get('_cargo_capture_table')
        if cache is not None and cache['capsules'] == capsules:
            return cache
        cells = [p for p in self.legal_cells if self._home(p)]
        index = {p: i for i, p in enumerate(cells)}
        size = len(cells)
        neighbors = [[index[q] for q in self.graph[p] if q in index] + [index[p]] for p in cells]
        # External neighbors and capsules are losing successors, so an evader
        # state with one of these replies can never be marked forced capture.
        remaining = [(len(self.graph[p]) + 1) for _ in cells for p in cells]
        defender, evader = [-1] * (size * size), [-1] * (size * size)
        queue = deque()
        for i, p in enumerate(cells):
            if p not in capsules:
                pair = i * size + i
                defender[pair] = evader[pair] = 0
                queue.append((0, pair))
                queue.append((1, pair))
        while queue:
            phase, pair = queue.popleft()
            ghost, enemy = divmod(pair, size)
            if phase == 1:
                for previous_ghost in neighbors[ghost]:
                    predecessor = previous_ghost * size + enemy
                    if defender[predecessor] < 0:
                        defender[predecessor] = evader[pair] + 1
                        queue.append((0, predecessor))
            else:
                for previous_enemy in neighbors[enemy]:
                    if cells[previous_enemy] in capsules:
                        continue
                    predecessor = ghost * size + previous_enemy
                    if evader[predecessor] >= 0:
                        continue
                    remaining[predecessor] -= 1
                    if remaining[predecessor] == 0:
                        evader[predecessor] = defender[pair] + 1
                        queue.append((1, predecessor))
        cache = {'capsules': capsules, 'index': index, 'cells': cells,
                 'size': size, 'defender': defender, 'evader': evader}
        self._team_memory['_cargo_capture_table'] = cache
        self._event('cargo_pursuit_table_built')
        return cache

    def _cargo_mission(self, state, tick):
        if self._intel_plan_cache is not None and self._intel_plan_cache[0] == tick:
            return self._intel_plan_cache[1]
        tracks = self._team_memory.get('_cargo_intelligence', {}).get('tracks', {})
        opponent_returned = sum(state.get_agent_state(i).num_returned for i in self.enemies)
        winning_load = max(1, self._intel_initial_food - 2 - opponent_returned)
        threats = [(i, track) for i, track in tracks.items()
                   if track.get('expected', 0) >= min(self.CARGO_MIN_LOAD, winning_load)
                   and state.get_agent_state(i).is_pacman]
        available = [(i, self._cell(state.get_agent_position(i))) for i in self.team
                     if not state.get_agent_state(i).is_pacman and
                     not state.get_agent_state(i).scared_timer]
        if not threats or not available:
            self._intel_plan_cache = tick, None
            return None
        table = self._cargo_capture_table(state)
        index, size = table['index'], table['size']
        options = []
        for enemy, track in threats:
            origins = sorted(track['positions'].items(), key=lambda item: -item[1])[:16]
            for teammate, position in available:
                if position not in index:
                    continue
                ghost = index[position]
                enemy_first = (enemy - self.index) % 4 < (teammate - self.index) % 4
                phase = table['evader'] if enemy_first else table['defender']
                coverage = sum(probability for origin, probability in origins if origin in index
                               and 0 <= phase[ghost * size + index[origin]] <= 24)
                if coverage < .9:
                    continue
                rank = sum(probability * phase[ghost * size + index[origin]]
                           for origin, probability in origins if origin in index
                           and 0 <= phase[ghost * size + index[origin]] <= 24) / coverage
                value = (track['expected'] + 2) * coverage / (1 + .18 * rank / 2)
                if teammate == min(self.team):
                    value -= .25
                if value <= 1:
                    continue
                # For another teammate, replan from its actual next observation.
                # Only the role reservation is used before its turn arrives.
                action, goal = Directions.STOP, position
                if teammate == self.index:
                    moves = []
                    for candidate in [position] + self.graph.get(position, []):
                        if candidate not in index:
                            continue
                        mass, expected_rank = 0.0, 0.0
                        for origin, probability in origins:
                            if origin not in index:
                                continue
                            successor = table['evader'][index[candidate] * size + index[origin]]
                            if 0 <= successor <= 24:
                                mass += probability
                                expected_rank += probability * successor
                        if mass >= .9:
                            moves.append((-expected_rank / mass, mass, candidate))
                    if not moves:
                        continue
                    _, _, goal = max(moves)
                    delta = goal[0] - position[0], goal[1] - position[1]
                    action = next(a for a, step in DIRECTION_STEP.items() if step == delta)
                options.append((value, teammate, enemy, goal, action, coverage, track['expected'], rank))
        result = None
        if options:
            value, teammate, enemy, goal, action, coverage, cargo, rank = max(options)
            result = {'agent': teammate, 'enemy': enemy, 'goal': goal, 'action': action,
                      'coverage': coverage, 'cargo': cargo, 'rank': rank, 'value': value, 'tick': tick}
        self._intel_plan_cache = tick, result
        return result

    def _choose_defense(self, state, pos, legal, threats, invaders, tick):
        plan = self._cargo_mission(state, tick)
        if plan and plan['agent'] == self.index and plan['action'] in legal:
            self._event('cargo_pursuit_turns')
            return plan['action'], plan['goal'], 'cargo-pursuit'
        return _BeforeCargoIntelligence._choose_defense(self, state, pos, legal, threats, invaders, tick)

# Five isolated teamwork experiments. All evidence comes from observations.
from collections import Counter

_TeamworkControl = CoordinatedAgent


class CoordinatedAgent(_TeamworkControl):
    TEAMWORK_FEATURES = ()

    def register_initial_state(self, state):
        super().register_initial_state(state)
        self._tw_history = deque(maxlen=24)
        self._tw_route = None
        self._tw_release_until = -1
        self._tw_release_enemy = None
        self._tw_last_request = -999
        self._tw_help = None
        self._tw_state = state
        self._current_tick = self._tick(state)

    def _tw_enabled(self, feature):
        return feature in self.TEAMWORK_FEATURES

    def _tw_action(self, origin, target):
        delta = target[0] - origin[0], target[1] - origin[1]
        return next((a for a, d in DIRECTION_STEP.items() if d == delta), Directions.STOP)

    def _tw_stalled(self, pos, carrying, returned):
        history = list(self._tw_history)[-8:]
        return (len(history) == 8 and carrying >= 3
                and all(r['cargo'] >= 3 and r['returned'] == returned
                        and r['kind'] in ('return', 'return-detour', 'capsule-rescue') for r in history)
                and len({r['pos'] for r in history} | {pos}) <= 4)

    def _update_beliefs(self, state, pos, tick):
        super()._update_beliefs(state, pos, tick)
        self._tw_state = state
        own = state.get_agent_state(self.index)
        stalled = self._tw_stalled(pos, own.num_carrying, own.num_returned)
        memory = self._team_memory.setdefault('_tw_status', {})
        memory[self.index] = {'tick': tick, 'pos': pos, 'cargo': own.num_carrying,
                             'returned': own.num_returned, 'stalled': stalled}
        if stalled and self._tw_enabled('assist'):
            request = self._team_memory.get('_tw_request', {})
            if request.get('agent') != self.index or request.get('until', -1) < tick:
                if tick - self._tw_last_request >= 64:
                    self._team_memory['_tw_request'] = {'agent': self.index, 'tick': tick,
                        'until': tick + 120, 'cargo': own.num_carrying,
                        'returned': own.num_returned, 'pos': pos, 'bank': self._return_portal,
                        'helper_crossed': False, 'guard_moved': False}
                    self._tw_last_request = tick
                    self._event('team_help_requested')
        request = self._team_memory.get('_tw_request', {})
        if request and request.get('until', -1) >= tick:
            requester = state.get_agent_state(request['agent'])
            if requester.num_returned > request['returned']:
                if request.get('helper_crossed'):
                    self._event('team_help_followed_by_bank')
                self._team_memory.pop('_tw_request', None)
            elif not requester.is_pacman or requester.num_carrying == 0:
                self._team_memory.pop('_tw_request', None)

    def _tw_loads(self, state):
        tracks = self._team_memory.get('_cargo_intelligence', {}).get('tracks', {})
        return {i: tracks.get(i, {}).get('expected', 0) for i in self.enemies
                if state.get_agent_state(i).is_pacman}

    def _tw_enemy_finish(self, state):
        returned = sum(state.get_agent_state(i).num_returned for i in self.enemies)
        remaining = max(1, self._intel_initial_food - 2 - returned)
        return any(load >= remaining for load in self._tw_loads(state).values())

    def _tw_path(self, start, targets, risk, penalties=None):
        penalties = penalties or {}
        todo, costs, parent = [(0.0, start)], {start: 0.0}, {}
        while todo:
            cost, cell = heapq.heappop(todo)
            if cost != costs[cell]:
                continue
            for nxt in self.graph[cell]:
                candidate = cost + 1 + risk.get(nxt, 0) + penalties.get(nxt, 0)
                if candidate < costs.get(nxt, float('inf')):
                    costs[nxt], parent[nxt] = candidate, cell
                    heapq.heappush(todo, (candidate, nxt))
        choices = [p for p in targets if p in costs and p != start]
        if not choices:
            return [], float('inf')
        goal = min(choices, key=lambda p: (costs[p], p))
        path = [goal]
        while path[-1] != start:
            path.append(parent[path[-1]])
        return list(reversed(path)), costs[goal]

    def _tw_ghosts(self, state):
        # Recent sightings remain estimates, never exact hidden configurations.
        ghosts = []
        tick = self._tick(state)
        for enemy in self.enemies:
            status = state.get_agent_state(enemy)
            if status.is_pacman:
                continue
            pos = self._cell(state.get_agent_position(enemy))
            if pos is not None:
                ghosts.append((enemy, pos, status.scared_timer))
            elif enemy in self.last_known_ghosts:
                old, when = self.last_known_ghosts[enemy]
                if tick - when <= 8:
                    ghosts.append((enemy, old, status.scared_timer))
        return ghosts

    def _return_route(self, pos, threats, carrying):
        normal = super()._return_route(pos, threats, carrying)
        if not self._tw_enabled('route') or not carrying:
            return normal
        state, tick = self._tw_state, self._current_tick
        own = state.get_agent_state(self.index)
        plan = self._tw_route
        if plan and (tick > plan['until'] or own.num_returned != plan['returned']):
            self._tw_route = plan = None
        if plan:
            path = plan['path']
            if pos in path:
                path = path[path.index(pos):]
                plan['path'] = path
            else:
                self._tw_route = plan = None
            if plan and len(path) > 1:
                nxt = path[1]
                ghosts = self._tw_ghosts(state)
                safe = (self._home(nxt) or nxt in self.get_capsules(state) or
                        all(timer > 1 or self._maze(nxt, ghost) > 1 for _, ghost, timer in ghosts))
                if safe:
                    self._event('return_committed_step')
                    return self._tw_action(pos, nxt), path[-1], 'return'
                self._tw_route = plan = None
        if not self._tw_stalled(pos, carrying, own.num_returned):
            return normal
        risk = self._danger_map(threats, carrying)
        # Stable recent sightings prevent a visibility boundary from flipping
        # the chosen path on alternating decisions.
        for _, ghost, timer in self._tw_ghosts(state):
            if timer <= 2:
                for cell in self.graph:
                    if not self._home(cell):
                        distance = self._maze(cell, ghost)
                        risk[cell] = max(risk.get(cell, 0), {0: 200, 1: 100, 2: 32, 3: 10, 4: 3}.get(distance, 0))
        repeated = Counter(r['pos'] for r in list(self._tw_history)[-8:])
        penalties = {p: 3 * n for p, n in repeated.items() if p != pos}
        path, cost = self._tw_path(pos, self.home_portals, risk, penalties)
        if len(path) < 2:
            return normal
        self._tw_route = {'path': path, 'until': tick + 4 * (len(path) + 8),
                          'returned': own.num_returned}
        self._return_portal = path[-1]
        self._event('return_committed_plan')
        return self._tw_action(pos, path[1]), path[-1], 'return'

    def _tw_help_plan(self, state, pos):
        if not self._tw_enabled('assist'):
            return None
        tick = self._tick(state)
        request = self._team_memory.get('_tw_request', {})
        own = state.get_agent_state(self.index)
        if (request.get('agent') != self.other or request.get('until', -1) < tick
                or own.num_carrying > 1 or self._tw_enemy_finish(state)):
            self._tw_help = None
            return None
        message = self._team_memory.get('_tw_status', {}).get(self.other, {})
        if tick - message.get('tick', -999) > 8 or message.get('cargo', 0) < 5:
            return None
        if self._tw_help and self._tw_help['request'] == request['tick']:
            target = self._tw_help['goal']
            if target != pos and target in self.graph:
                return target, self._tw_help['kind']
            self._tw_help = None
        mate = self._cell(state.get_agent_position(self.other))
        bank = request.get('bank') or min(self.home_portals, key=lambda p: self._maze(mate, p))
        ghosts = self._tw_ghosts(state)
        active = [p for _, p, timer in ghosts if timer <= 2]
        # Prefer a reachable capsule. It changes the guard's capabilities even
        # when the guard refuses to chase the helper.
        capsules = self.get_capsules(state)
        candidates = []
        for capsule in capsules:
            travel = self._maze(pos, capsule)
            if travel + self._maze(mate, bank) <= max(0, state.data.timeleft // 4) + 5:
                candidates.append((travel + .15 * self._maze(mate, capsule), capsule, 'team-help-capsule'))
        if candidates:
            _, target, kind = min(candidates)
        else:
            portals = [p for p in self.enemy_portals if self._maze(p, bank) >= 5
                       and all(self._maze(p, guard) >= 3 for guard in active)]
            if not portals:
                return None
            target = min(portals, key=lambda p: self._maze(pos, p) + .2 * self._maze(mate, p))
            kind = 'team-help-pressure'
        self._tw_help = {'request': request['tick'], 'goal': target, 'kind': kind}
        request['helper'] = self.index
        self._event('team_help_started')
        return target, kind

    def _tw_stale_defense(self, state, pos, invaders):
        history = list(self._tw_history)[-8:]
        if len(history) < 8 or self._tw_enemy_finish(state):
            return False
        if not all(r['mode'] == 'defend' and r['kind'] == 'reactive-capsule-cutoff'
                   for r in history):
            return False
        if len({r['pos'] for r in history} | {pos}) > 2:
            return False
        for enemy, enemy_pos, status in invaders:
            seen = [r['enemies'].get(enemy) for r in history]
            if (all(p is not None for p in seen) and len(set(seen + [enemy_pos])) <= 2
                    and status.num_carrying < 3 and self._maze(pos, enemy_pos) > 1):
                return True
        return False

    def _tw_futile_chase(self, state, pos, invaders):
        history = list(self._tw_history)[-5:]
        if len(history) < 5 or not all(r['mode'] == 'defend' and r['kind'] == 'intercept' for r in history):
            return None
        for enemy, enemy_pos, status in invaders:
            old = [r['enemies'].get(enemy) for r in history]
            if not all(p is not None for p in old) or self._maze(pos, enemy_pos) <= 1:
                continue
            gaps = [self._maze(r['pos'], p) for r, p in zip(history, old)] + [self._maze(pos, enemy_pos)]
            exits = [min(self._maze(p, gate) for gate in self.home_portals) for p in old + [enemy_pos]]
            committed = (all(b < a for a, b in zip(exits, exits[1:])) and exits[-1] <= 8
                         and all(r['loads'].get(enemy) == status.num_carrying for r in history))
            if not committed or gaps[-1] < gaps[0]:
                continue
            table = self._cargo_capture_table(state)
            index, size = table['index'], table['size']
            if pos not in index or enemy_pos not in index:
                continue
            rank = table['defender'][index[pos] * size + index[enemy_pos]]
            if rank >= 0:
                continue
            distances, _, _ = self._intel_home_routes(pos)
            nearest = min(self._maze(enemy_pos, gate) for gate in self.home_portals)
            feasible = [gate for gate in self.home_portals
                        if self._maze(enemy_pos, gate) <= nearest + 2
                        and distances.get(gate, 999) + 1 <= self._maze(enemy_pos, gate)]
            if not feasible:
                return enemy
        return None

    def _should_defend(self, state, pos, invaders, tick):
        own = state.get_agent_state(self.index)
        if not own.num_carrying and not own.scared_timer:
            if self._tw_help_plan(state, pos):
                self._event('team_help_role_turns')
                return False
            if self._tw_enabled('defense'):
                if self._tw_stale_defense(state, pos, invaders):
                    self._tw_release_until = tick + 48
                    self._tw_release_enemy = None
                    self._event('stale_defense_released')
                if tick < self._tw_release_until and self._tw_release_enemy is None:
                    if not self._tw_enemy_finish(state) and max(self._tw_loads(state).values(), default=0) < 5:
                        return False
            if self._tw_enabled('pursuit'):
                enemy = self._tw_futile_chase(state, pos, invaders)
                if enemy is not None:
                    self._tw_release_enemy = enemy
                    self._tw_release_until = tick + 24
                    self._event('futile_chase_released')
                if tick < self._tw_release_until and self._tw_release_enemy is not None:
                    others = [i for i in self._tw_loads(state) if i != self._tw_release_enemy]
                    if not others and all(self._maze(pos, p) > 1 for _, p, _ in invaders):
                        return False
        return super()._should_defend(state, pos, invaders, tick)

    def _choose_attack(self, state, pos, legal, threats, carrying, mate):
        help_plan = self._tw_help_plan(state, pos)
        if help_plan:
            goal, kind = help_plan
            risk = self._danger_map(threats, carrying)
            risk[goal] = 0 if kind == 'team-help-capsule' else risk.get(goal, 0)
            costs, actions = self._routes(pos, risk)
            if goal in actions and actions[goal] in legal:
                request = self._team_memory.get('_tw_request', {})
                if self._enemy_home(pos) and not request.get('helper_crossed'):
                    request['helper_crossed'] = True
                    self._event('team_help_crossed')
                if kind == 'team-help-capsule' and self._maze(pos, goal) == 1:
                    self._event('team_help_capsule_reached')
                return actions[goal], goal, kind
        if self._tw_enabled('safety') and carrying:
            returned = sum(state.get_agent_state(i).num_returned for i in self.team)
            turns = state.data.timeleft // 4
            closing = returned + carrying >= self.returned_to_win
            urgent = self._tw_enemy_finish(state) and self.get_score(state) + carrying > 0
            if closing or urgent or turns <= self._bank_distance(pos) + 8:
                self._event('finish_delivery_turns')
                return self._return_route(pos, threats, carrying)
        return super()._choose_attack(state, pos, legal, threats, carrying, mate)

    def _bank_target(self, state):
        normal = super()._bank_target(state)
        if self._tw_enabled('safety'):
            returned = sum(state.get_agent_state(i).num_returned for i in self.team)
            return min(normal, max(1, self.returned_to_win - returned))
        return normal

    def _best_food(self, state, pos, costs, food, carrying, mate, live_threats):
        if not self._tw_enabled('safety'):
            return super()._best_food(state, pos, costs, food, carrying, mate, live_threats)
        ghosts = self._tw_ghosts(state)
        safe = []
        for target in food:
            mouth = self.trap_mouth.get(target)
            trip = self._maze(pos, target) + (self._maze(target, mouth) if mouth else 0)
            blocked = mouth is not None and any(
                timer <= trip + 2 and self._maze(guard, mouth) <= trip + 1 for _, guard, timer in ghosts)
            if not blocked:
                safe.append(target)
            else:
                self._event('unsafe_food_trip_rejected')
        returned = sum(state.get_agent_state(i).num_returned for i in self.team)
        near_finish = self.returned_to_win - returned - carrying <= 3 or self._tw_enemy_finish(state)
        if near_finish:
            targets = [(costs[target] + self._bank_distance(target), target) for target in safe if target in costs]
            if targets:
                cost, target = min(targets)
                self._event('finish_roundtrip_target')
                return target, 13 - cost
        return super()._best_food(state, pos, costs, safe, carrying, mate, live_threats)

    def _tactical_action(self, state, pos, legal, threats, carrying, normal):
        normal = super()._tactical_action(state, pos, legal, threats, carrying, normal)
        if not self._tw_enabled('safety') or not self._enemy_home(pos):
            return normal
        ghosts = [(p, timer) for _, p, timer in self._tw_ghosts(state)
                  if timer <= 3 and self._maze(pos, p) <= 7]
        if not ghosts:
            return normal
        capsules = set(self.get_capsules(state))
        action, goal, kind = normal
        horizon = 4

        def survival(cell, depth, visited):
            if self._home(cell) or cell in capsules:
                return horizon + 2
            # Conservative arrival envelope, not a claim about hidden actions.
            if any(timer <= depth and self._maze(cell, guard) <= depth for guard, timer in ghosts):
                return depth - 1
            if depth == horizon:
                return horizon
            choices = [p for p in self.graph[cell] if p not in visited]
            return max((survival(p, depth + 1, visited | {p}) for p in choices), default=depth)

        choices = []
        for candidate in legal:
            dx, dy = DIRECTION_STEP[candidate]
            nxt = pos[0] + dx, pos[1] + dy
            if nxt not in self.graph:
                continue
            depth = survival(nxt, 1, {pos, nxt})
            value = (depth, -self._maze(nxt, goal) - .2 * self.recent.count(nxt), candidate)
            choices.append(value)
        current = next((item for item in choices if item[2] == action), None)
        best = max(choices, default=None)
        if current and best and best[0] >= current[0] + 2:
            self._event('early_corridor_escape')
            return best[2], goal, kind
        return normal

    def choose_action(self, state):
        action = super().choose_action(state)
        own = state.get_agent_state(self.index)
        self._tw_history.append({'pos': self._cell(state.get_agent_position(self.index)),
            'cargo': own.num_carrying, 'returned': own.num_returned,
            'mode': self.last_mode, 'kind': self.goal_kind,
            'enemies': {i: self._cell(state.get_agent_position(i)) for i in self.enemies},
            'loads': {i: state.get_agent_state(i).num_carrying for i in self.enemies
                      if state.get_agent_position(i) is not None}})
        return action

CoordinatedAgent.TEAMWORK_FEATURES = ('route',)

"""Four independent, observation-only improvements on the frozen V6."""
_DarioControl = CoordinatedAgent


class CoordinatedAgent(_DarioControl):
    IMPROVEMENT_FEATURES = ()

    def register_initial_state(self, state):
        super().register_initial_state(state)
        self._dv_history = deque(maxlen=32)
        self._dv_break_plan = None
        self._dv_break_after = -1
        self._dv_avoid = {}
        self._dv_defense_until = -1
        self._dv_capsules = set(self.get_capsules(state))
        self._dv_power_pending = False
        self._dv_power_plan = None
        food = len(self.get_food(state).as_list()) + len(self.get_food_you_are_defending(state).as_list())
        # The native engine uses total_food / 2 - 2 for both teams, even on
        # a slightly asymmetric supplied board.
        self._dv_win = max(1, math.ceil(food / 2 - 2))

    def _dv_on(self, feature):
        return feature in self.IMPROVEMENT_FEATURES

    def _update_beliefs(self, state, pos, tick):
        super()._update_beliefs(state, pos, tick)
        capsules = set(self.get_capsules(state))
        if self._dv_capsules - capsules:
            self._dv_power_pending = True
            self._dv_power_plan = None
        self._dv_capsules = capsules
        self._dv_avoid = {p: until for p, until in self._dv_avoid.items() if until >= tick}

    def _dv_loads(self, state):
        tracks = self._team_memory.get('_cargo_intelligence', {}).get('tracks', {})
        return {i: (state.get_agent_state(i).num_carrying if state.get_agent_position(i) is not None
                    else tracks.get(i, {}).get('upper', 0))
                for i in self.enemies if state.get_agent_state(i).is_pacman}

    def _dv_decisive(self, state):
        needed = self._dv_win - sum(state.get_agent_state(i).num_returned for i in self.enemies)
        return max(self._dv_loads(state).values(), default=0) >= max(1, needed)

    def _dv_deadline(self, state):
        """Optimistic enemy finish ETA, in own rounds; an estimate, not truth."""
        if not self._dv_on('deadline'):
            return None
        needed = self._dv_win - sum(state.get_agent_state(i).num_returned for i in self.enemies)
        if needed > 12 or not self.enemy_portals:
            return None
        tracks = self._team_memory.get('_cargo_intelligence', {}).get('tracks', {})
        food = set(self.get_food_you_are_defending(state).as_list())
        estimates = []
        for enemy in self.enemies:
            if not state.get_agent_state(enemy).is_pacman:
                continue
            track = tracks.get(enemy, {})
            visible = self._cell(state.get_agent_position(enemy))
            cargo = (state.get_agent_state(enemy).num_carrying if visible is not None
                     else track.get('expected', 0))
            shortfall = max(0, math.ceil(needed - cargo))
            if shortfall > 6 or shortfall > len(food):
                continue
            origins = [(visible, 1)] if visible is not None else sorted(
                track.get('positions', {}).items(), key=lambda item: -item[1])[:6]
            for origin, probability in origins:
                if probability < .05:
                    continue
                here, travel, available = origin, 0, set(food)
                for _ in range(shortfall):
                    target = min(available, key=lambda p: (self._maze(here, p), p))
                    travel += self._maze(here, target)
                    here = target
                    available.remove(target)
                travel += min(self._maze(here, gate) for gate in self.enemy_portals)
                estimates.append(travel + 2)
        return min(estimates) if estimates else None

    def _dv_empty_stall(self, state, pos):
        own = state.get_agent_state(self.index)
        rows = list(self._dv_history)[-12:]
        return (not own.num_carrying and len(rows) == 12
                and all(r['mode'] == 'attack' and r['cargo'] == 0
                        and r['kind'] in ('food', 'probe', 'capsule') for r in rows)
                and len({r['pos'] for r in rows} | {pos}) <= 6
                and len({(r['returned'], r['food']) for r in rows}) == 1)

    def _dv_first_safe(self, state, pos, nxt):
        return (self._home(nxt) or nxt in self.get_capsules(state)
                or all(timer > 1 or self._maze(nxt, guard) > 1
                       for _, guard, timer in self._tw_ghosts(state)))

    def _dv_breakout(self, state, pos, legal, threats, mate):
        if not self._dv_on('breakout') or state.get_agent_state(self.index).num_carrying:
            self._dv_break_plan = None
            return None
        tick = self._tick(state)
        plan = self._dv_break_plan
        if plan:
            if tick > plan['until'] or pos not in plan['path'] or len(self.get_food(state).as_list()) < plan['food']:
                self._dv_break_plan = plan = None
            else:
                path = plan['path'][plan['path'].index(pos):]
                if len(path) > 1 and self._dv_first_safe(state, pos, path[1]):
                    plan['path'] = path
                    self._event('empty_breakout_step')
                    return self._tw_action(pos, path[1]), path[-1], 'empty-breakout'
                self._dv_break_plan = plan = None
        if tick < self._dv_break_after or not self._dv_empty_stall(state, pos):
            return None
        old = self.goal
        if old is not None:
            self._dv_avoid[old] = tick + 120
        risk = self._danger_map(threats, 0)
        repeated = Counter(r['pos'] for r in list(self._dv_history)[-12:])
        penalties = {p: 2 * n for p, n in repeated.items() if p != pos}
        entry = min(self.enemy_portals, key=lambda p: self._maze(pos, p)) if self.enemy_portals else pos
        guards = [p for _, p, timer in self._tw_ghosts(state) if timer <= 2]
        gates = [p for p in self.enemy_portals if self._maze(entry, p) >= 4
                 and all(self._maze(p, guard) >= 3 for guard in guards)]
        goals = gates + [c for c in self.get_capsules(state) if self._maze(pos, c) <= 20]
        path, _ = self._tw_path(pos, goals, risk, penalties)
        self._dv_break_after = tick + 48
        if len(path) < 2 or not self._dv_first_safe(state, pos, path[1]):
            self._event('empty_breakout_no_safe_route')
            return None
        self._dv_break_plan = {'path': path, 'until': tick + 4 * (len(path) + 5),
                               'food': len(self.get_food(state).as_list())}
        self._team_memory['_dv_pressure'] = {'requester': self.index, 'until': tick + 96,
                                              'target': path[-1], 'origin': pos}
        self._event('empty_breakout_started')
        return self._tw_action(pos, path[1]), path[-1], 'empty-breakout'

    def _dv_help_goal(self, state, pos):
        if not self._dv_on('breakout'):
            return None
        request = self._team_memory.get('_dv_pressure', {})
        own = state.get_agent_state(self.index)
        if (request.get('requester') != self.other or request.get('until', -1) < self._tick(state)
                or own.num_carrying or self._dv_decisive(state)
                or max(self._dv_loads(state).values(), default=0) >= 3):
            return None
        if any(state.get_agent_position(i) is not None
               and state.get_agent_state(i).is_pacman
               and self._maze(pos, self._cell(state.get_agent_position(i))) <= 1 for i in self.enemies):
            return None
        targets = list(self.get_capsules(state))
        kind = 'empty-help-capsule'
        if not targets:
            targets = [p for p in self.enemy_portals if self._maze(p, request['target']) >= 4]
            kind = 'empty-help-pressure'
        return (min(targets, key=lambda p: (self._maze(pos, p), p)), kind) if targets else None

    def _dv_stale_defense(self, state, pos):
        rows = list(self._dv_history)[-12:]
        if (len(rows) < 12 or self._dv_decisive(state)
                or max(self._dv_loads(state).values(), default=0) >= 3
                or (self.get_score(state) > 0 and state.data.timeleft < 240)):
            return False
        kinds = ('reactive-food-cutoff', 'reactive-exit-cutoff', 'reactive-capsule-cutoff')
        if not all(r['mode'] == 'defend' and r['kind'] in kinds for r in rows):
            return False
        if len({r['pos'] for r in rows} | {pos}) > 2 or len({r['defended'] for r in rows}) != 1:
            return False
        for enemy in self.enemies:
            visible = self._cell(state.get_agent_position(enemy))
            if visible is not None and state.get_agent_state(enemy).is_pacman and self._maze(pos, visible) <= 1:
                return False
        return True

    def _should_defend(self, state, pos, invaders, tick):
        own = state.get_agent_state(self.index)
        if not own.num_carrying and not own.scared_timer:
            if self._dv_help_goal(state, pos):
                self._event('empty_help_role')
                return False
            if self._dv_on('defense'):
                if self._dv_stale_defense(state, pos):
                    self._dv_defense_until = tick + 80
                    self._event('unproductive_blockade_released')
                adjacent = any(self._maze(pos, p) <= 1 for _, p, _ in invaders)
                if (tick < self._dv_defense_until and not adjacent and not self._dv_decisive(state)
                        and max(self._dv_loads(state).values(), default=0) < 3):
                    self._event('blockade_release_attack_turn')
                    return False
        return super()._should_defend(state, pos, invaders, tick)

    def _dv_build_power(self, state, pos, budget):
        food = set(self.get_food(state).as_list())
        if not food or not self.home_portals:
            return None
        best = None
        for first in sorted(food, key=lambda p: (self._maze(pos, p), p))[:5]:
            here, elapsed, order, available = pos, 0, [], set(food)
            target = first
            while target is not None:
                travel = self._maze(here, target)
                if elapsed + travel + self._bank_distance(target) > budget:
                    break
                elapsed += travel
                order.append(target)
                available.remove(target)
                here = target
                feasible = [p for p in available if elapsed + self._maze(here, p) + self._bank_distance(p) <= budget]
                target = min(feasible, key=lambda p: (self._maze(here, p), self._bank_distance(p), p)) if feasible else None
            if not order:
                continue
            bank = min(self.home_portals, key=lambda p: (self._maze(here, p), p))
            candidate = (len(order), -elapsed - self._maze(here, bank), order, bank)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            return None
        return {'targets': best[2], 'bank': best[3]}

    def _dv_power_action(self, state, pos, legal, threats, carrying):
        if not self._dv_on('capsule'):
            return None
        own = state.get_agent_state(self.index)
        timer = min(state.get_agent_state(i).scared_timer for i in self.enemies)
        if not own.is_pacman or timer < 4:
            self._dv_power_plan = None
            self._dv_power_pending = False
            return None
        if self._dv_power_pending:
            self._dv_power_pending = False
            budget = min(timer - 3, max(0, state.data.timeleft // 4 - 2))
            deadline = self._dv_deadline(state)
            if deadline is not None:
                budget = min(budget, max(0, deadline - 2))
            self._dv_power_plan = self._dv_build_power(state, pos, budget)
            if self._dv_power_plan:
                self._event('power_harvest_plan_created')
        plan = self._dv_power_plan
        if not plan:
            return None
        food = set(self.get_food(state).as_list())
        plan['targets'] = [p for p in plan['targets'] if p in food and p != pos]
        returned = sum(state.get_agent_state(i).num_returned for i in self.team)
        if carrying and (timer <= self._bank_distance(pos) + 3
                         or returned + carrying >= self._dv_win or not plan['targets']):
            self._event('power_harvest_return')
            return self._return_route(pos, threats, carrying)
        if not plan['targets']:
            self._dv_power_plan = None
            return None
        goal = plan['targets'][0]
        if self._maze(pos, goal) + self._bank_distance(goal) > timer - 3:
            self._dv_power_plan = None
            return self._return_route(pos, threats, carrying) if carrying else None
        costs, actions = self._routes(pos, self._danger_map(threats, carrying))
        action = actions.get(goal)
        if action in legal:
            self._event('power_harvest_step')
            return action, goal, 'power-harvest'
        self._dv_power_plan = None
        return None

    def _best_food(self, state, pos, costs, food, carrying, mate, live_threats):
        if self._dv_on('breakout') and self._dv_avoid:
            different = [p for p in food if all(self._maze(p, q) >= 4 for q in self._dv_avoid)]
            if different:
                food = different
        deadline = self._dv_deadline(state)
        if deadline is not None:
            feasible = [p for p in food if self._maze(pos, p) + self._bank_distance(p) + 2 <= deadline]
            if feasible:
                self._event('finish_roundtrip_food')
                food = feasible
        return super()._best_food(state, pos, costs, food, carrying, mate, live_threats)

    def _choose_attack(self, state, pos, legal, threats, carrying, mate):
        deadline = self._dv_deadline(state)
        if deadline is not None and carrying and deadline <= self._bank_distance(pos) + 8:
            self._event('enemy_finish_bank_early')
            return self._return_route(pos, threats, carrying)
        power = self._dv_power_action(state, pos, legal, threats, carrying)
        if power:
            return power
        help_goal = self._dv_help_goal(state, pos)
        if help_goal:
            goal, kind = help_goal
            _, actions = self._routes(pos, self._danger_map(threats, carrying))
            action = actions.get(goal)
            if action in legal:
                dx, dy = DIRECTION_STEP[action]
                if self._dv_first_safe(state, pos, (pos[0] + dx, pos[1] + dy)):
                    self._event('empty_help_step')
                    return action, goal, kind
        breakout = self._dv_breakout(state, pos, legal, threats, mate)
        return breakout if breakout else super()._choose_attack(state, pos, legal, threats, carrying, mate)

    def choose_action(self, state):
        action = super().choose_action(state)
        own = state.get_agent_state(self.index)
        self._dv_history.append({'pos': self._cell(state.get_agent_position(self.index)),
            'cargo': own.num_carrying, 'returned': own.num_returned, 'kind': self.goal_kind,
            'mode': self.last_mode, 'food': len(self.get_food(state).as_list()),
            'defended': len(self.get_food_you_are_defending(state).as_list())})
        return action

CoordinatedAgent.IMPROVEMENT_FEATURES = ('deadline',)


# Online route search and a behavior-based option controller. No opponent IDs.
_R26V6 = CoordinatedAgent
R26_CONFIG = {'starts': 4, 'local': 16, 'replan': 0, 'weight': 0.15, 'search': 'insert', 'gate': 'always', 'shield': True}


class CoordinatedAgent(_R26V6):
    def register_initial_state(self, state):
        super().register_initial_state(state)
        self._r26_cfg = dict(R26_CONFIG)
        self._r26_dist = {}

    def _r26_d(self, a, b):
        key = (a,b)
        if key not in self._r26_dist:
            self._r26_dist[key] = self._r26_dist[b,a] = self._maze(a,b)
        return self._r26_dist[key]

    def _update_beliefs(self, state, pos, tick):
        super()._update_beliefs(state,pos,tick)
        memory = self._team_memory
        profile = memory.setdefault('_r26_profile', {'tick':-1,'dual':None,'armed':False,
            'guard_until':-1,'small_banks':0,'returned':{},'food_start':len(self.get_food_you_are_defending(state).as_list()),
            'last_visible':{}})
        if profile['tick']==tick:
            return
        both = all(state.get_agent_state(i).is_pacman for i in self.enemies)
        if both:
            if profile['dual'] is None:
                profile['dual'] = tick
        else:
            profile['dual'] = None
        for enemy in self.enemies:
            status = state.get_agent_state(enemy)
            returned = status.num_returned
            old = profile['returned'].get(enemy,returned)
            if 0<returned-old<=6:
                profile['small_banks'] += 1
                profile['armed'] = False
                self._event('r26_small_bank_disarm')
            profile['returned'][enemy] = returned
            location = self._cell(state.get_agent_position(enemy))
            if location is None:
                continue
            old_seen = profile['last_visible'].get(enemy)
            if not status.is_pacman and status.scared_timer<=2:
                attackers = [self._cell(state.get_agent_position(i)) for i in self.team
                             if state.get_agent_state(i).is_pacman]
                distances = [self._maze(location,p) for p in attackers]
                closing = (old_seen and tick-old_seen[0]<=8 and attackers
                           and min(distances)<min(self._maze(old_seen[1],p) for p in attackers))
                if distances and (min(distances)<=4 or (closing and min(distances)<=7)):
                    profile['guard_until'] = tick+64
                    profile['armed'] = False
                    self._event('r26_guard_disarm')
            profile['last_visible'][enemy] = (tick,location)
        tracks = memory.get('_cargo_intelligence',{}).get('tracks',{})
        lower = [state.get_agent_state(i).num_carrying if state.get_agent_position(i) is not None
                 else tracks.get(i,{}).get('lower',0) for i in self.enemies]
        eaten = profile['food_start']-len(self.get_food_you_are_defending(state).as_list())
        delay = 8 if self._r26_cfg['gate']=='fast' else 16
        evidence = min(lower)>=2 if self._r26_cfg['gate']=='cautious' else (min(lower)>=1 or eaten>=6)
        if (both and tick-profile['dual']>=delay and evidence and profile['small_banks']==0
                and tick>profile['guard_until'] and not profile['armed']):
            profile['armed'] = True
            memory.pop('_r26_plan',None)
            self._event('r26_option_armed')
        profile['tick'] = tick

    def _r26_active(self, state):
        if self._r26_cfg['gate']=='off' or len(self.get_food(state).as_list())>64:
            return False
        if self._r26_cfg['gate']=='always':
            return True
        p = self._team_memory.get('_r26_profile',{})
        return p.get('armed',False) and self._tick(state)>p.get('guard_until',-1) and not p.get('small_banks',0)

    def _r26_critical(self, state):
        need = max(1,self._dv_win-sum(state.get_agent_state(i).num_returned for i in self.enemies))
        tracks = self._team_memory.get('_cargo_intelligence',{}).get('tracks',{})
        loads = [state.get_agent_state(i).num_carrying if state.get_agent_position(i) is not None
                 else tracks.get(i,{}).get('lower',0) for i in self.enemies
                 if state.get_agent_state(i).is_pacman]
        return sum(loads)>=need

    def _should_defend(self, state, pos, invaders, tick):
        if not self._r26_active(state):
            return super()._should_defend(state,pos,invaders,tick)
        own = state.get_agent_state(self.index)
        if not own.is_pacman and not own.scared_timer:
            caps = self.get_capsules_you_are_defending(state)
            if any(self._maze(pos,p)<=1 and p not in caps for _,p,_ in invaders):
                return True
            if self._r26_critical(state):
                return super()._should_defend(state,pos,invaders,tick)
            if self._r26_cfg['shield']:
                mission = self._cargo_mission(state,tick)
                if mission and mission['agent']==self.index:
                    self._event('r26_proven_defense_kept')
                    return True
                if any(self._maze(pos,p)<=3 for _,p,_ in invaders) and caps:
                    self._event('r26_capsule_defense_kept')
                    return super()._should_defend(state,pos,invaders,tick)
        self._event('r26_race_role')
        return False

    def _cooperative_attack(self, state, pos, legal, threats, carrying, mate, normal):
        if self._r26_active(state):
            return normal
        return super()._cooperative_attack(state,pos,legal,threats,carrying,mate,normal)

    def _r26_make_plan(self, state):
        agents = sorted(self.team)
        starts = [self._cell(state.get_agent_position(i)) for i in agents]
        food = sorted(self.get_food(state).as_list())
        if len(food)>64 or not food:
            return None
        returned = sum(state.get_agent_state(i).num_returned for i in agents)
        cargo = sum(state.get_agent_state(i).num_carrying for i in agents)
        quota = max(0,min(len(food),self._dv_win-returned-cargo))
        d = self._r26_d
        bank = {p:self._bank_distance(p) for p in food+starts}
        w = self._r26_cfg['weight']
        phase = [(i-self.index)%4/4 for i in agents]
        def objective(lengths):
            a,b = lengths[0]+phase[0],lengths[1]+phase[1]
            return max(a,b)+w*(a+b)
        def length(route,k):
            path = [starts[k]]+route
            return sum(d(a,b) for a,b in zip(path,path[1:]))+bank[path[-1]]
        def insertion(route,k,target,j):
            prev = route[j-1] if j else starts[k]
            nxt = route[j] if j<len(route) else None
            return d(prev,target)+(d(target,nxt) if nxt else bank[target])-(d(prev,nxt) if nxt else bank[prev])
        def improve(routes):
            lengths = [length(routes[k],k) for k in (0,1)]
            for _ in range(self._r26_cfg['local']):
                old,winner = objective(lengths),None
                for k in (0,1):
                    for i in range(len(routes[k])):
                        for j in range(i+2,len(routes[k])+1):
                            a = routes[k][i-1] if i else starts[k]
                            b,c = routes[k][i],routes[k][j-1]
                            e = routes[k][j] if j<len(routes[k]) else None
                            delta = d(a,c)-d(a,b)+(d(b,e)-d(c,e) if e else bank[b]-bank[c])
                            trial = list(lengths);trial[k]+=delta
                            value = objective(trial)
                            if value<old-.001:
                                old,winner = value,('reverse',k,i,j,trial)
                        reduced = routes[k][:i]+routes[k][i+1:]
                        reduced_len = length(reduced,k)
                        target = routes[k][i]
                        for j in range(len(routes[1-k])+1):
                            trial = list(lengths);trial[k]=reduced_len
                            trial[1-k]+=insertion(routes[1-k],1-k,target,j)
                            value = objective(trial)
                            if value<old-.001:
                                old,winner = value,('move',k,i,j,trial)
                if winner is None:
                    break
                kind,k,i,j,lengths = winner
                if kind=='reverse':
                    routes[k][i:j] = reversed(routes[k][i:j])
                else:
                    routes[1-k].insert(j,routes[k].pop(i))
            return objective(lengths),routes
        candidates = []
        for seed in range(self._r26_cfg['starts']):
            routes,remaining = [[],[]],set(food)
            lengths = [bank[p] for p in starts]
            # Distinct deterministic initial centers avoid one insertion basin.
            if seed and quota>=2:
                first = min(food,key=lambda p:(d(starts[0],p)+(seed%2)*.35*p[1],p))
                second = min((p for p in food if p!=first),key=lambda p:(d(starts[1],p)-.25*seed*d(first,p),p))
                routes = [[first],[second]]
                remaining.difference_update((first,second))
                lengths = [length(routes[k],k) for k in (0,1)]
            while remaining and sum(map(len,routes))<quota:
                best = None
                for target in sorted(remaining):
                    for k in (0,1):
                        for j in range(len(routes[k])+1):
                            trial = list(lengths);trial[k]+=insertion(routes[k],k,target,j)
                            value = objective(trial)
                            # Alternative seeds encourage different early partitions.
                            if seed:
                                value += .04*seed*((target[1]+k*7)%5)
                            proposal = (value,target,k,j,trial)
                            if best is None or proposal[:4]<best[:4]:
                                best = proposal
                _,target,k,j,lengths = best
                routes[k].insert(j,target);remaining.remove(target)
            candidates.append(improve(routes))
        if self._r26_cfg['search']=='beam' and quota:
            # Bounded joint best-first beam: endpoint states share one food mask.
            beam = [(0,([[],[]],[bank[p] for p in starts],frozenset(food)))]
            for depth in range(quota):
                following = []
                for _,(routes,lengths,remaining) in beam:
                    for k in (0,1):
                        here = routes[k][-1] if routes[k] else starts[k]
                        options = sorted(remaining,key=lambda p:(d(here,p),bank[p],p))[:4]
                        for target in options:
                            trial = list(lengths);trial[k]+=d(here,target)+bank[target]-bank[here]
                            rr = [list(route) for route in routes];rr[k].append(target)
                            pending = remaining-{target}
                            lower = (quota-depth-1)*.65
                            score = objective(trial)+lower
                            following.append((score,(rr,trial,pending)))
                following.sort(key=lambda p:(p[0],p[1][0]))
                beam = following[:24]
            candidates.extend(improve(routes) for _,(routes,_,_) in beam[:3])
        value,routes = min(candidates,key=lambda x:(x[0],x[1]))
        plan = {'routes':dict(zip(agents,routes)),'food':set(food),'returned':returned,
                'tick':self._tick(state),'value':value,'starts':dict(zip(agents,starts))}
        self._team_memory['_r26_plan'] = plan
        self._event('r26_planned')
        return plan

    def _choose_attack(self, state, pos, legal, threats, carrying, mate):
        if not self._r26_active(state):
            return super()._choose_attack(state,pos,legal,threats,carrying,mate)
        ghosts = [p for i,p,s in threats if s.scared_timer<=2 and state.get_agent_position(i) is not None]
        if ghosts and min(self._maze(pos,p) for p in ghosts)<=7:
            self._event('r26_tactical_fallback')
            return super()._choose_attack(state,pos,legal,threats,carrying,mate)
        returned = sum(state.get_agent_state(i).num_returned for i in self.team)
        cargo = sum(state.get_agent_state(i).num_carrying for i in self.team)
        if carrying and (returned+cargo>=self._dv_win or state.data.timeleft//4<=self._bank_distance(pos)+3):
            self._event('r26_winning_return')
            return self._return_route(pos,threats,carrying)
        food = set(self.get_food(state).as_list())
        plan = self._team_memory.get('_r26_plan')
        replan = self._r26_cfg['replan']
        if (plan is None or food-plan['food'] or returned!=plan['returned']
                or (replan and len(plan['food']-food)>=replan)
                or (not carrying and pos==self.start and self._tick(state)-plan['tick']>160)):
            plan = self._r26_make_plan(state)
        if not plan:
            return super()._choose_attack(state,pos,legal,threats,carrying,mate)
        route = plan['routes'][self.index]
        route[:] = [p for p in route if p in food]
        if not route:
            if carrying:
                self._event('r26_route_return')
                return self._return_route(pos,threats,carrying)
            plan = self._r26_make_plan(state)
            route = plan['routes'][self.index] if plan else []
        if route:
            goal = route[0]
            _,actions = self._routes(pos,self._danger_map(threats,carrying))
            action = actions.get(goal)
            if action in legal:
                self._event('r26_route_step')
                return action,goal,'r26-route'
        return super()._choose_attack(state,pos,legal,threats,carrying,mate)


# A stateful option controller; identities and repository names are not inputs.
_R26Route = CoordinatedAgent
R26_META = {'start': False, 'food': 0, 'small': 8, 'closing': 2, 'protect': True}


class CoordinatedAgent(_R26Route):
    def register_initial_state(self, state):
        super().register_initial_state(state)
        self._r26_meta = dict(R26_META)
        food = self.get_food_you_are_defending(state).as_list()
        initial_eta = {i:4*min((self._maze(self._cell(state.get_initial_agent_position(i)),p)
                               for p in food),default=20)+80 for i in self.enemies}
        self._team_memory.setdefault('_r26_meta',{'tick':-1,'active':self._r26_meta['start'],
            'ever':self._r26_meta['start'],'guard_until':-1,'small_banks':0,'food_start':len(food),
            'seen_raid':set(),'returned':{},'visible':{},'closing':{},'arrival':initial_eta})

    def _update_beliefs(self, state, pos, tick):
        # Update the original V6 ledger exactly once, without the rejected
        # prototype classifier's blanket guard proximity / late food gate.
        _R26V6._update_beliefs(self,state,pos,tick)
        m = self._team_memory['_r26_meta']
        if m['tick']==tick:
            return
        previous = m['active']
        for enemy in self.enemies:
            status = state.get_agent_state(enemy)
            if status.is_pacman:
                m['seen_raid'].add(enemy)
                m['closing'][enemy]=0
            returned = status.num_returned
            amount = returned-m['returned'].get(enemy,returned)
            if self._r26_meta['small'] and 0<amount<=self._r26_meta['small']:
                m['small_banks']+=1
                self._event('r26_meta_small_deposit')
            m['returned'][enemy]=returned
            visible = self._cell(state.get_agent_position(enemy))
            old = m['visible'].get(enemy)
            if visible is not None:
                attackers = [self._cell(state.get_agent_position(i)) for i in self.team
                             if state.get_agent_state(i).is_pacman]
                approaching = (not status.is_pacman and status.scared_timer<=2 and old
                    and tick-old[0]<=8 and attackers and min(self._maze(visible,p) for p in attackers)<=6
                    and min(self._maze(visible,p) for p in attackers)<min(self._maze(old[1],p) for p in attackers))
                if approaching:
                    m['closing'][enemy]=m['closing'].get(enemy,0)+1
                    if m['closing'][enemy]>=self._r26_meta['closing']:
                        m['guard_until']=tick+96
                        self._event('r26_meta_pursuing_guard')
                elif not status.is_pacman:
                    m['closing'][enemy]=max(0,m['closing'].get(enemy,0)-1)
                m['visible'][enemy]=(tick,visible)
        patrol = any(i not in m['seen_raid'] and tick>m['arrival'][i] for i in self.enemies)
        blocked = patrol or m['small_banks']>0 or tick<=m['guard_until']
        both = all(state.get_agent_state(i).is_pacman for i in self.enemies)
        eaten = m['food_start']-len(self.get_food_you_are_defending(state).as_list())
        if blocked:
            m['active']=False
        elif both and eaten>=self._r26_meta['food']:
            m['active']=True
            m['ever']=True
        if previous!=m['active']:
            self._team_memory.pop('_r26_plan',None)
            self._event('r26_meta_on' if m['active'] else 'r26_meta_off')
            if not m['active'] and self.goal_kind=='r26-route':
                self.goal=self.goal_kind=None
        m['tick']=tick

    def _r26_active(self, state):
        return (len(self.get_food(state).as_list())<=64
                and self._team_memory.get('_r26_meta',{}).get('active',False))

    def _should_defend(self, state, pos, invaders, tick):
        own = state.get_agent_state(self.index)
        if (self._r26_meta['protect'] and self._r26_active(state) and not own.is_pacman
                and not own.scared_timer and invaders
                and _R26V6._should_defend(self,state,pos,invaders,tick)):
            threats,_ = self._enemy_threat(state,tick)
            action,goal,kind = _R26V6._choose_defense(self,state,pos,state.get_legal_actions(self.index),threats,invaders,tick)
            if kind=='reactive-capsule-cutoff' and self._maze(pos,goal)<=1:
                self._event('r26_meta_capsule_gate_kept')
                return True
        return super()._should_defend(state,pos,invaders,tick)

    def _choose_attack(self, state, pos, legal, threats, carrying, mate):
        m = self._team_memory.get('_r26_meta',{})
        returned = sum(state.get_agent_state(i).num_returned for i in self.team)
        cargo = sum(state.get_agent_state(i).num_carrying for i in self.team)
        if m.get('ever') and carrying and returned+cargo>=self._dv_win:
            self._event('r26_meta_finish_committed')
            return self._return_route(pos,threats,carrying)
        return super()._choose_attack(state,pos,legal,threats,carrying,mate)


"""V9: observation inventory, food recovery and guarded death placement.

Append to the frozen V8 source to make a standalone contest submission.
No background work, hidden configurations, observer feedback or file IO.
"""

_V9Base = CoordinatedAgent
_V9Factory = create_team


def create_team(first_index, second_index, is_red,
                first='CoordinatedAgent', second='CoordinatedAgent',
                num_training=0, strategy=None, cargo_mode='api',
                recovery=True, sacrifice=True):
    if cargo_mode not in ('api', 'observed'):
        raise ValueError('cargo_mode must be api or observed')
    def enabled(value):
        if isinstance(value, str):
            if value.lower() not in ('true', 'false', '1', '0'):
                raise ValueError('Expected true/false option')
            return value.lower() in ('true', '1')
        return bool(value)
    team = _V9Factory(first_index, second_index, is_red, first, second,
                      num_training, strategy)
    for agent in team:
        agent.v9_cargo_mode = cargo_mode
        agent.v9_recovery = enabled(recovery)
        agent.v9_sacrifice = enabled(sacrifice)
    return team


class CoordinatedAgent(_V9Base):
    v9_cargo_mode = 'api'
    v9_recovery = True
    v9_sacrifice = True

    def register_initial_state(self, state):
        super().register_initial_state(state)
        # Both initial registrations precede play. Reset all per-game memory.
        self._team_memory['_v9'] = {
            'tick': None, 'attack_food': set(self.get_food(state).as_list()),
            'defend_food': set(self.get_food_you_are_defending(state).as_list()),
            'attack_drops': {}, 'defend_drops': {}}
        self._v9_history = deque(maxlen=12)

    def _update_cargo_intelligence(self, state, tick):
        super()._update_cargo_intelligence(state, tick)
        if self.v9_cargo_mode != 'api':
            return
        ledger = self._team_memory['_cargo_intelligence']
        loads = []
        for enemy in self.enemies:
            status = state.get_agent_state(enemy)
            amount = getattr(status, 'num_carrying', None)
            if not isinstance(amount, (int, float)) or amount < 0:
                # A future engine may redact this field. Keep the V8 inference.
                return
            amount = int(amount) if status.is_pacman else 0
            loads.append(amount)
        for enemy, amount in zip(self.enemies, loads):
            record = ledger['tracks'][enemy]
            record.update(lower=amount, upper=amount, expected=float(amount))
        ledger.update(total=sum(loads), allocations=[tuple(loads)])
        self._intel_plan_cache = None
        self._event('v9_api_inventory')

    def _update_beliefs(self, state, pos, tick):
        super()._update_beliefs(state, pos, tick)
        memory = self._team_memory['_v9']
        if memory['tick'] == tick:
            return
        for side, current in (
                ('attack', set(self.get_food(state).as_list())),
                ('defend', set(self.get_food_you_are_defending(state).as_list()))):
            previous = memory[side + '_food']
            recent = memory[side + '_drops']
            for cell in current - previous:
                recent[cell] = tick + 160
            memory[side + '_drops'] = {
                cell: until for cell, until in recent.items()
                if cell in current and until >= tick}
            memory[side + '_food'] = current
            if current - previous:
                self._event('v9_' + side + '_food_restored')
        memory['tick'] = tick

    def _dv_deadline(self, state):
        normal = super()._dv_deadline(state)
        if not self._dv_on('deadline'):
            return normal
        needed = max(1, self._dv_win - sum(state.get_agent_state(i).num_returned for i in self.enemies))
        tracks = self._team_memory.get('_cargo_intelligence', {}).get('tracks', {})
        arrivals = []
        for enemy in self.enemies:
            if not state.get_agent_state(enemy).is_pacman:
                continue
            record = tracks.get(enemy, {})
            amount = record.get('lower', 0)
            if amount <= 0:
                continue
            visible = self._cell(state.get_agent_position(enemy))
            origins = [(visible, 1.0)] if visible is not None else list(record.get('positions', {}).items())
            eta = min((min((self._maze(p, gate) for gate in self.enemy_portals), default=999)
                       for p, probability in origins if probability >= .05), default=999)
            if eta < 999:
                arrivals.append((eta + 2, amount))
        # A large individual or joint return can end the game even when the
        # older food-collection estimate's twelve-pellet cutoff is exceeded.
        estimates = [normal] if normal is not None else []
        for eta, amount in arrivals:
            if amount >= needed:
                estimates.append(eta)
        if arrivals and sum(amount for _, amount in arrivals) >= needed:
            estimates.append(max(eta for eta, _ in arrivals))
        return min(estimates) if estimates else None

    def _cargo_destinations(self, state, origin, track):
        normal = super()._cargo_destinations(state, origin, track)
        if not self.v9_recovery or track.get('expected', 0) >= 3:
            return normal
        drops = self._team_memory['_v9']['defend_drops']
        choices = sorted(drops, key=lambda p: (self._maze(origin, p), p))[:3]
        if not choices or self._maze(origin, choices[0]) > 12:
            return normal
        # A possible return to the dropped cluster, never a claimed position.
        weighted = {p: .8 * weight for p, weight in normal}
        weights = [1.0 / (1 + self._maze(origin, p)) for p in choices]
        for p, weight in zip(choices, weights):
            weighted[p] = weighted.get(p, 0) + .2 * weight / sum(weights)
        total = sum(weighted.values())
        return [(p, w / total) for p, w in sorted(weighted.items())]

    def _defense_goal(self, state, pos, invaders, tick):
        normal = super()._defense_goal(state, pos, invaders, tick)
        own = state.get_agent_state(self.index)
        if (not self.v9_recovery or invaders or own.scared_timer or own.is_pacman
                or normal[1] != 'patrol'):
            return normal
        drops = self._team_memory['_v9']['defend_drops']
        choices = []
        for enemy in self.enemies:
            if not state.get_agent_state(enemy).is_pacman:
                continue
            distribution = self.beliefs.get(enemy, {})
            for target in drops:
                eta = self._maze(pos, target)
                if eta > 8:
                    continue
                probability = sum(w for p, w in distribution.items()
                                  if self._maze(p, target) <= eta + 3)
                if probability >= .65:
                    choices.append((probability - .025 * eta, target))
        if choices:
            self._event('v9_recovery_patrol')
            return max(choices)[1], 'v9-recovery-patrol'
        return normal

    def _v9_live_guards(self, state):
        return [(i, self._cell(state.get_agent_position(i)), state.get_agent_state(i))
                for i in self.enemies
                if state.get_agent_position(i) is not None
                and not state.get_agent_state(i).is_pacman
                and state.get_agent_state(i).scared_timer <= 1]

    def _v9_safe_paths(self, state, start):
        """Conservative moving-guard reachability, with teammate overlap allowed.

        Guard distance is measured on corridors; each future round expands its
        possible reach by one. Absence of a path is not proof of forced death.
        """
        guards = self._v9_live_guards(state)
        capsules = set(self.get_capsules(state))
        distance, first = {start: 0}, {start: Directions.STOP}
        todo = deque([start])
        while todo:
            p = todo.popleft()
            # Reaching a bank/capsule is a safe terminal, not a tunnel through it.
            if p != start and ((not self._home(start) and self._home(p)) or p in capsules):
                continue
            for nxt in self.graph.get(p, ()):
                depth = distance[p] + 1
                if nxt in distance:
                    continue
                terminal = self._home(nxt) or nxt in capsules
                if not terminal and any(self._maze(nxt, g) <= depth for _, g, _ in guards):
                    continue
                # On the first step even a capsule/bank must be reached from
                # a living position; consume/type change precedes collision.
                distance[nxt] = depth
                delta = nxt[0] - p[0], nxt[1] - p[1]
                first[nxt] = (next(a for a, d in DIRECTION_STEP.items() if d == delta)
                              if p == start else first[p])
                todo.append(nxt)
        return distance, first

    def _v9_predict_dump(self, state, death, amount, victim=None, consumed=False,
                         occupant_updates=None):
        """Mirror native eight-neighbor BFS; walls block placement, not expansion.

        Unseen occupants remain unknown. Use exact native successors for an
        immediate sacrifice; this predictor is for visible/hypothetical states.
        """
        victim = self.index if victim is None else victim
        food = set(state.get_red_food().as_list()) | set(state.get_blue_food().as_list())
        if consumed:
            food.discard(death)
        capsules = set(state.get_capsules())
        occupied = {i: self._cell(state.get_agent_position(i)) for i in self.team + self.enemies}
        occupied[victim] = death
        occupied.update(occupant_updates or {})
        occupied = set(occupied.values())
        home_side = self._home(death)
        todo, seen, found = deque([death]), set(), []
        # All native legal placements lie in this finite box. Expanding one
        # extra outer ring preserves the unbounded native BFS's relative order.
        while todo and len(found) < amount:
            p = todo.popleft()
            if p in seen:
                continue
            seen.add(p)
            x, y = p
            if (p in self.graph and 0 < x < self.width and 0 < y < self.height
                    and self._home(p) == home_side and p not in food
                    and p not in capsules and p not in occupied):
                found.append(p)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nxt = x + dx, y + dy
                    if (-1 <= nxt[0] <= self.width and -1 <= nxt[1] <= self.height
                            and nxt not in seen):
                        todo.append(nxt)
        return found if len(found) == amount else []

    def _v9_recovery_choice(self, state, pos, legal, threats, carrying, normal):
        if not self.v9_recovery or carrying or normal[2] not in ('food', 'probe', 'r26-route'):
            return None
        food = set(self.get_food(state).as_list())
        drops = self._team_memory['_v9']['attack_drops'].keys() & food
        if not drops:
            return None
        costs, first = self._v9_safe_paths(state, pos)
        normal_goal = normal[1]
        if normal_goal is None:
            return None
        ordinary = costs.get(normal_goal, 9999) + self._bank_distance(normal_goal)
        mate = self._cell(state.get_agent_position(self.other))
        message = self._team_memory.get(self.other, {})
        choices = []
        for target in sorted(drops):
            if target not in costs or first[target] not in legal:
                continue
            total = costs[target] + self._bank_distance(target)
            if total + 3 >= ordinary or total + 4 >= state.data.timeleft // 4:
                continue
            # Reserve food, never floor tiles: sharing a corridor is legal.
            if (mate is not None and message.get('goal') == target
                    and self._tick(state) - message.get('tick', -999) <= 8
                    and self._maze(mate, target) <= costs[target]):
                continue
            choices.append((total, target))
        if choices:
            _, goal = min(choices)
            self._event('v9_recovery_step')
            return first[goal], goal, 'v9-recover'
        return None

    def _v9_drop_quality(self, state, drop, guards):
        """Cost of actual collection + delivery, not wall count/Manhattan range."""
        mate = self._cell(state.get_agent_position(self.other))
        runners = [self.start]
        if mate is not None and state.get_agent_state(self.other).num_carrying <= 2:
            runners.append(mate)
        values = []
        for start in runners:
            remaining, here, length = set(drop), start, 0
            while remaining:
                target = min(remaining, key=lambda p: (self._maze(here, p), p))
                length += self._maze(here, target)
                here = target
                remaining.remove(target)
            length += self._bank_distance(here)
            values.append(length + (1 if start == self.start else 0))
        # Narrow branches and nearby defenders can make a short trip unsafe.
        danger = sum(max(0, 4 - min((self._maze(p, g) for _, g, _ in guards), default=99))
                     + .3 * min(8, self.trap_depth.get(p, 0)) for p in drop)
        return min(values, default=9999) + danger

    def _v9_sacrifice_choice(self, state, pos, legal, carrying):
        own = state.get_agent_state(self.index)
        if (not self.v9_sacrifice or not own.is_pacman or not 1 <= carrying <= 3
                or self.get_score(state) > 0 or self._dv_decisive(state)
                or sum(state.get_agent_state(i).num_returned + state.get_agent_state(i).num_carrying
                       for i in self.team) >= self._dv_win):
            return None
        rows = list(self._v9_history)[-8:]
        if (len(rows) < 8 or any(r['cargo'] != carrying for r in rows)
                or len({r['pos'] for r in rows} | {pos}) > 2):
            return None
        paths, _ = self._v9_safe_paths(state, pos)
        if any(p in paths for p in self.home_portals + list(self.get_capsules(state))):
            return None
        mate = self._cell(state.get_agent_position(self.other))
        if mate is None or state.get_agent_state(self.other).num_carrying > 2:
            return None
        mate_paths, _ = self._v9_safe_paths(state, mate)
        guards = self._v9_live_guards(state)
        candidates = []
        for action in legal:
            dx, dy = DIRECTION_STEP[action]
            death = pos[0] + dx, pos[1] + dy
            if death in self.get_capsules(state) or not any(g == death and s.scared_timer == 0 for _, g, s in guards):
                continue
            # Native successor preserves capsule-before-collision and occupancy,
            # including simultaneous co-located teammates. No hidden state added.
            after = state.generate_successor(self.index, action)
            if after.get_agent_state(self.index).num_carrying:
                continue
            drop = sorted(set(self.get_food(after).as_list()) - set(self.get_food(state).as_list()))
            amount = carrying + int(death in self.get_food(state).as_list())
            if len(drop) != amount:
                continue
            # Unknown enemies cannot occupy the drop if every selected square
            # lies in the team's current sight radius.
            if any(min(self._manhattan(p, pos), self._manhattan(p, mate)) > 5 for p in drop):
                continue
            # Require a complete safe pickup AND a way back from every pellet.
            if not all(p in mate_paths and mate_paths[p] + self._bank_distance(p) <= 10 for p in drop):
                continue
            recovery = self._v9_drop_quality(state, drop, guards)
            if recovery + 5 >= state.data.timeleft // 4:
                continue
            candidates.append((recovery, action, death))
        if candidates:
            _, action, death = min(candidates)
            self._event('v9_guarded_sacrifice')
            return action, death, 'v9-sacrifice'
        return None

    def _choose_attack(self, state, pos, legal, threats, carrying, mate):
        normal = super()._choose_attack(state, pos, legal, threats, carrying, mate)
        recovery = self._v9_recovery_choice(state, pos, legal, threats, carrying, normal)
        sacrifice = self._v9_sacrifice_choice(state, pos, legal, carrying)
        return sacrifice or recovery or normal

    def _tactical_action(self, state, pos, legal, threats, carrying, normal):
        if normal[2] == 'v9-sacrifice':
            return normal
        return super()._tactical_action(state, pos, legal, threats, carrying, normal)

    def choose_action(self, state):
        action = super().choose_action(state)
        self._v9_history.append({'pos': self._cell(state.get_agent_position(self.index)),
                                 'cargo': state.get_agent_state(self.index).num_carrying})
        return action


"""Shared within-game evidence and conservative empirical route risk.

Append to standalone V9. All updates run synchronously on our own callbacks.
Only exact single-move sightings become movement samples. Missing sightings
remain missing; compact observations/events are retained for this game only.
"""
_HistoryBase = CoordinatedAgent


class CoordinatedAgent(_HistoryBase):
    HISTORY_LEARNING = True

    def register_initial_state(self, state):
        super().register_initial_state(state)
        self._history_spawns = {i: self._cell(state.get_initial_agent_position(i)) for i in self.enemies}
        self._team_memory['_history_model'] = {
            'tick': None, 'observer': None, 'previous': None, 'log': [],
            'enemies': {i: {'chase': [0, 0], 'retreat': [0, 0], 'stay': [0, 0]}
                        for i in self.enemies},
            'routes': {}, 'episodes': {}, 'entries': {}, 'events': []}

    def _history_snapshot(self, state):
        # Enemy inventory is intentionally omitted: observed mode retains its
        # original contract even if the engine leaves hidden payloads present.
        return {i: {'pos': self._cell(state.get_agent_position(i)),
                    'pacman': bool(state.get_agent_state(i).is_pacman),
                    'scared': state.get_agent_state(i).scared_timer,
                    'cargo': state.get_agent_state(i).num_carrying if i in self.team else None,
                    'returned': state.get_agent_state(i).num_returned}
                for i in self.team + self.enemies}

    def _history_region(self, pos):
        # Geometry is used for travel. The bucket pools a few related entry
        # rows only for statistics; it never replaces maze distances.
        portal = min(self.home_portals, key=lambda p: (self._maze(pos, p), p), default=self.start)
        return portal[1] // 4

    def _history_movements(self, memory, snapshot, tick):
        previous = memory['previous']
        if previous is None:
            return
        elapsed = tick - memory['tick']
        for enemy in self.enemies:
            before, after = memory.get('after_own', previous)[enemy], snapshot[enemy]
            moves = sum((memory['observer'] + k) % 4 == enemy for k in range(elapsed))
            origin, target = before['pos'], after['pos']
            if (moves != 1 or origin is None or target is None
                    or before['pacman'] != after['pacman']
                    or self._manhattan(origin, target) > 1):
                continue
            # A captured ghost can jump back to a nearby spawn. Such a reset
            # must not be mislabelled as a chosen direction.
            if (target == self._history_spawns[enemy]
                    and origin != target):
                continue
            stats = memory['enemies'][enemy]
            stats['stay'][0] += int(origin == target)
            stats['stay'][1] += 1
            # Positions of our agents after our preceding actual chosen move
            # are known, giving the opponent's pre-move target configuration.
            teammates = memory.get('after_own', previous)
            targets = [s['pos'] for i, s in teammates.items() if i in self.team
                       and s['pos'] is not None and s['pacman']]
            if not after['pacman'] and before['scared'] == 0 and targets:
                old = min(self._maze(origin, p) for p in targets)
                new = min(self._maze(target, p) for p in targets)
                if old <= 8:
                    stats['chase'][0] += int(new < old)
                    stats['chase'][1] += 1
            elif after['pacman']:
                old = min((self._maze(origin, p) for p in self.enemy_portals), default=999)
                new = min((self._maze(target, p) for p in self.enemy_portals), default=999)
                stats['retreat'][0] += int(new < old)
                stats['retreat'][1] += 1
            memory['events'].append({'tick': tick, 'enemy': enemy, 'type': 'exact_move',
                                     'from': origin, 'to': target})
            self._event('h91_exact_move_sample')

    def _history_outcomes(self, memory, snapshot, tick):
        previous = memory['previous']
        for index in self.team:
            current = snapshot[index]
            old = previous[index] if previous else current
            if current['pacman'] and not old['pacman']:
                memory['entries'][index] = self._history_region(current['pos'])
            episode = memory['episodes'].get(index)
            if episode is None and current['cargo'] > 0:
                region = memory['entries'].get(index, self._history_region(current['pos']))
                episode = {'region': region, 'max_cargo': current['cargo']}
                memory['episodes'][index] = episode
            if episode is None:
                continue
            episode['max_cargo'] = max(episode['max_cargo'], current['cargo'])
            delivered = current['returned'] - old['returned']
            lost = old['cargo'] > 0 and current['cargo'] == 0 and not delivered and not current['pacman']
            if delivered > 0 or lost:
                route = memory['routes'].setdefault(episode['region'],
                    {'banked_trips': 0, 'lost_trips': 0, 'banked_food': 0, 'lost_food': 0})
                if delivered > 0:
                    route['banked_trips'] += 1
                    route['banked_food'] += delivered
                else:
                    route['lost_trips'] += 1
                    route['lost_food'] += old['cargo']
                memory['events'].append({'tick': tick, 'agent': index,
                    'type': 'bank' if delivered > 0 else 'cargo_loss',
                    'region': episode['region'], 'food': delivered or old['cargo']})
                self._event('h91_trip_banked' if delivered > 0 else 'h91_trip_lost')
                del memory['episodes'][index]

    def _history_update(self, state, tick):
        memory = self._team_memory['_history_model']
        if memory['tick'] == tick:
            return
        snapshot = self._history_snapshot(state)
        self._history_movements(memory, snapshot, tick)
        self._history_outcomes(memory, snapshot, tick)
        memory['log'].append({'tick': tick, 'observer': self.index, 'agents': snapshot})
        memory.update(tick=tick, observer=self.index,
                      previous={i: dict(value) for i, value in snapshot.items()})
        memory.pop('after_own', None)

    def _update_beliefs(self, state, pos, tick):
        super()._update_beliefs(state, pos, tick)
        if self.HISTORY_LEARNING:
            self._history_update(state, tick)

    def _history_failure(self, region):
        stats = self._team_memory['_history_model']['routes'].get(region)
        if not stats:
            return None
        total = stats['banked_trips'] + stats['lost_trips']
        if total < 3:
            return None
        # Beta(1,4) failure prior: a few observations cannot instantly turn a
        # route into an impassable obstacle or a claimed success guarantee.
        return (stats['lost_trips'] + 1.0) / (total + 5.0)

    def _danger_map(self, threats, carrying, scared_self=0):
        risk = super()._danger_map(threats, carrying, scared_self)
        if not self.HISTORY_LEARNING or '_history_model' not in self._team_memory:
            return risk
        model = self._team_memory['_history_model']
        for enemy, position, status in threats:
            success, samples = model['enemies'][enemy]['chase']
            if samples < 6 or status.scared_timer > 1:
                continue
            chase = (success + 2.0) / (samples + 4.0)
            if chase > .6:
                for p in tuple(risk):
                    if self._maze(p, position) <= 4:
                        risk[p] *= 1 + min(.5, chase - .6)
                self._event('h91_chase_risk_used')
        # Repeated failed raids can alter portal choice even after a defender
        # disappears. Finite costs allow reconsideration after circumstances change.
        for portal in self.enemy_portals:
            failure = self._history_failure(portal[1] // 4)
            if failure is not None and failure > .35:
                risk[portal] = risk.get(portal, 0) + min(4, (1 + carrying * .15) * failure * 2)
                self._event('h91_route_risk_used')
        return risk

    def _v9_sacrifice_choice(self, state, pos, legal, carrying):
        if self.HISTORY_LEARNING:
            failure = self._history_failure(self._history_region(pos))
            if failure is not None and failure >= .5:
                self._event('h91_repeated_failure_reset_rejected')
                return None
        return super()._v9_sacrifice_choice(state, pos, legal, carrying)

    def choose_action(self, state):
        action = super().choose_action(state)
        if self.HISTORY_LEARNING:
            # This is a prediction of OUR immediately executing known action,
            # not a guessed opponent move. Native visible collision rules apply.
            after = state.generate_successor(self.index, action)
            snapshot = self._history_snapshot(after)
            memory = self._team_memory['_history_model']
            self._history_outcomes(memory, snapshot, self._tick(state))
            memory['after_own'] = snapshot
            # Avoid counting a deposit/loss again at the next team's observation.
            for index in self.team:
                memory['previous'][index] = snapshot[index]
        return action

    def final(self, state):
        # final() receives an unfiltered state in this engine. Only own inventory
        # is used to close terminal trips; no final enemy positions are recorded.
        if self.HISTORY_LEARNING:
            memory = self._team_memory['_history_model']
            if memory['previous'] is not None:
                snapshot = dict(memory['previous'])
                for index in self.team:
                    status = state.get_agent_state(index)
                    snapshot[index] = {'pos': self._cell(state.get_agent_position(index)),
                        'pacman': status.is_pacman, 'scared': status.scared_timer,
                        'cargo': status.num_carrying, 'returned': status.num_returned}
                self._history_outcomes(memory, snapshot, self._tick(state))
                memory['previous'] = snapshot
        super().final(state)


"""Gradual within-game route evidence, discounted by current guard context.

Append to frozen V9.1. No training state survives register_initial_state().
Live pressure is not counted as another completed raid or cargo loss.
"""
_H92Base = CoordinatedAgent


class CoordinatedAgent(_H92Base):
    HISTORY_CONTEXT = True

    def register_initial_state(self, state):
        super().register_initial_state(state)
        self._h92_state = None
        self._team_memory['_history_context'] = {'guards': {}, 'raids': {}}

    def _history_failure(self, region):
        stats = self._team_memory['_history_model']['routes'].get(region)
        if not stats:
            return None
        losses = stats.get('context_lost_weight', stats['lost_trips']) if self.HISTORY_CONTEXT else stats['lost_trips']
        return (losses + 1.0) / (stats['banked_trips'] + losses + 5.0)

    def _h92_route_penalty(self, region):
        failure = self._history_failure(region)
        return 0.0 if failure is None else min(1.0, 2.0 * max(0.0, failure - .20))

    def _history_outcomes(self, memory, snapshot, tick):
        begin = len(memory['events'])
        previous = memory['previous']
        super()._history_outcomes(memory, snapshot, tick)
        if not self.HISTORY_CONTEXT:
            return
        for event in memory['events'][begin:]:
            if event['type'] != 'cargo_loss':
                continue
            region = event['region']
            # Last observed pre-capture position is evidence, not an exact
            # hidden death location. Distant losses weakly implicate entry.
            location = previous[event['agent']]['pos'] if previous else None
            gates = [p for p in self.enemy_portals if p[1] // 4 == region]
            distance = min((self._maze(location, p) for p in gates), default=999) if location else 999
            weight = 1.0 if distance <= 8 else .25
            stats = memory['routes'][region]
            stats['context_lost_weight'] = stats.get('context_lost_weight', 0.0) + weight
            event['entry_loss_weight'] = weight
            self._event('h92_local_loss' if weight == 1 else 'h92_distant_loss_discounted')

    def _history_update(self, state, tick):
        already = self._team_memory['_history_model']['tick'] == tick
        super()._history_update(state, tick)
        self._h92_state = state
        if already or not self.HISTORY_CONTEXT:
            return
        context = self._team_memory['_history_context']
        for enemy in self.enemies:
            position = self._cell(state.get_agent_position(enemy))
            status = state.get_agent_state(enemy)
            if status.is_pacman or status.scared_timer > 2:
                context['guards'].pop(enemy, None)
            elif position is not None:
                context['guards'][enemy] = {'pos': position, 'tick': tick}
        context['guards'] = {i: record for i, record in context['guards'].items()
                             if tick - record['tick'] <= 16}
        for index in self.team:
            status = state.get_agent_state(index)
            if not status.is_pacman:
                context['raids'].pop(index, None)
                continue
            pos = self._cell(state.get_agent_position(index))
            raid = context['raids'].setdefault(index, {'signals': set(), 'pos': pos})
            visible = [(i, self._cell(state.get_agent_position(i))) for i in self.enemies
                       if state.get_agent_position(i) is not None
                       and not state.get_agent_state(i).is_pacman
                       and state.get_agent_state(i).scared_timer <= 2]
            nearby = any(self._maze(pos, p) <= 5 for _, p in visible)
            bank = self._bank_distance(pos)
            exits = [p for p in self.home_portals if self._maze(pos, p) <= bank + 1]
            contested = bool(exits) and all(any(self._maze(g, p) <= self._maze(pos, p)
                                                for _, g in visible) for p in exits)
            turned = (nearby and raid['pos'] != pos
                      and self._bank_distance(pos) < self._bank_distance(raid['pos']))
            for name, present in (('guard_pressure', nearby), ('exit_contested', contested),
                                  ('pressure_turnback', turned)):
                if present and name not in raid['signals']:
                    raid['signals'].add(name)
                    self._event('h92_' + name)
            raid['pos'] = pos

    def _h92_portal_context(self, portal):
        """Return (history multiplier, immediate pressure), both bounded.

        Only visible or at most 16-global-move-old observed guard positions
        contribute. Hidden positions are never treated as exact new samples.
        """
        state = self._h92_state
        if state is None:
            return .25, 0.0
        tick = self._tick(state)
        pressure = 0.0
        for record in self._team_memory['_history_context']['guards'].values():
            age = max(0, tick - record['tick'])
            if age > 16:
                continue
            distance = self._maze(portal, record['pos'])
            proximity = max(0.0, 1.0 - distance / 7.0)
            pressure = max(pressure, proximity * (1.0 - age / 20.0))
        # A current opportunity weakens old evidence; it never makes the
        # complete route 'safe'. Native live danger retains priority.
        opportunity = all(state.get_agent_state(i).is_pacman
                          or state.get_agent_state(i).scared_timer > 2 for i in self.enemies)
        return ((.1 if opportunity else .25) + .75 * pressure), pressure

    def _danger_map(self, threats, carrying, scared_self=0):
        # _HistoryBase is frozen V9: bypass only V9.1's abrupt route penalty.
        # Keep V9.1's learned chasing rule unchanged for the controlled ablation.
        risk = _HistoryBase._danger_map(self, threats, carrying, scared_self)
        if not self.HISTORY_LEARNING or '_history_model' not in self._team_memory:
            return risk
        model = self._team_memory['_history_model']
        for enemy, position, status in threats:
            successes, samples = model['enemies'][enemy]['chase']
            if samples < 6 or status.scared_timer > 1:
                continue
            chase = (successes + 2.0) / (samples + 4.0)
            if chase > .6:
                for cell in tuple(risk):
                    if self._maze(cell, position) <= 4:
                        risk[cell] *= 1 + min(.5, chase - .6)
                self._event('h91_chase_risk_used')
        for portal in self.enemy_portals:
            penalty = self._h92_route_penalty(portal[1] // 4)
            multiplier, pressure = self._h92_portal_context(portal) if self.HISTORY_CONTEXT else (1.0, 0.0)
            addition = min(1.0, penalty * multiplier + .6 * pressure)
            if addition > 0:
                risk[portal] = risk.get(portal, 0.0) + addition
                self._event('h92_route_risk_used' if penalty else 'h92_live_portal_risk_used')
        return risk


"""Delivery commitments and local, within-game chase evidence for V9.2.

Append to the frozen 2026-11-07 V9.2. No files, clocks or background work
are used by the policy. Teammate positions remain traversable.
"""
_Delivery92Base = CoordinatedAgent


class CoordinatedAgent(_Delivery92Base):
    def register_initial_state(self, state):
        super().register_initial_state(state)
        self._delivery_plan = None
        self._team_memory['_delivery92'] = {'chase': {}, 'intents': {}}

    @staticmethod
    def _delivery_bucket(distance, cargo):
        return ('near' if distance <= 4 else 'far', 'loaded' if cargo >= 4 else 'light')

    def _history_movements(self, memory, snapshot, tick):
        previous = memory['previous']
        super()._history_movements(memory, snapshot, tick)
        if previous is None:
            return
        before_all = memory.get('after_own', previous)
        elapsed = tick - memory['tick']
        for enemy in self.enemies:
            before, after = before_all[enemy], snapshot[enemy]
            origin, target = before['pos'], after['pos']
            moves = sum((memory['observer'] + k) % 4 == enemy for k in range(elapsed))
            if (moves != 1 or origin is None or target is None
                    or before['pacman'] or after['pacman'] or before['scared']
                    or self._manhattan(origin, target) > 1
                    or target == self._history_spawns[enemy] and origin != target):
                continue
            candidates = [(self._maze(origin, s['pos']), i, s) for i, s in before_all.items()
                          if i in self.team and s['pacman'] and s['pos'] is not None]
            if not candidates:
                continue
            distance, _, victim = min(candidates, key=lambda item: item[:2])
            if distance > 8 or not any(self._maze(p, victim['pos']) < distance
                                       for p in self.graph[origin]):
                continue  # No opportunity to approach: no negative chase sample.
            bucket = (enemy,) + self._delivery_bucket(distance, victim['cargo'])
            stats = self._team_memory['_delivery92']['chase'].setdefault(bucket, [0, 0])
            stats[0] += int(self._maze(target, victim['pos']) < distance)
            stats[1] += 1
            self._event('delivery_context_sample')

    def _danger_map(self, threats, carrying, scared_self=0):
        # Use physical V9 danger; replace the inherited global chase statistic.
        risk = _HistoryBase._danger_map(self, threats, carrying, scared_self)
        state = self._h92_state
        if not self.HISTORY_LEARNING:
            return risk
        own = self._cell(state.get_agent_position(self.index)) if state else None
        for enemy, position, status in threats:
            visible = self._cell(state.get_agent_position(enemy)) if state else None
            if visible is None or status.is_pacman or status.scared_timer > 1:
                continue
            distance = self._maze(own, visible)
            if distance > 8:
                continue
            bucket = (enemy,) + self._delivery_bucket(distance, carrying)
            success, samples = self._team_memory['_delivery92']['chase'].get(bucket, (0, 0))
            # Shrink sparse evidence toward neutral and never lower physical danger.
            probability = (success + 2.0) / (samples + 4.0)
            addition = min(.25, max(0, probability - .5) * samples / (samples + 8.0))
            if addition:
                for cell in tuple(risk):
                    if self._maze(cell, visible) <= 4:
                        risk[cell] *= 1 + addition
                self._event('delivery_context_risk')
        for portal in self.enemy_portals:
            penalty = self._h92_route_penalty(portal[1] // 4)
            multiplier, pressure = self._h92_portal_context(portal) if self.HISTORY_CONTEXT else (1.0, 0.0)
            addition = min(1.0, penalty * multiplier + .6 * pressure)
            if addition:
                risk[portal] = risk.get(portal, 0) + addition
                self._event('delivery_portal_risk')
        # Native consumption happens before collision at the capsule itself.
        # Cells leading TO a capsule retain their danger.
        for capsule in self.get_capsules(state) if state else ():
            risk[self._cell(capsule)] = 0.0
        return risk

    def _delivery_paths(self, start, risk):
        costs, parent = {start: 0.0}, {}
        queue = [(0.0, start)]
        while queue:
            cost, pos = heapq.heappop(queue)
            if cost != costs[pos]:
                continue
            for nxt in self.graph[pos]:
                value = cost + 1 + risk.get(nxt, 0)
                if value < costs.get(nxt, float('inf')):
                    costs[nxt], parent[nxt] = value, pos
                    heapq.heappush(queue, (value, nxt))
        paths = {}
        for goal in self.home_portals:
            if goal not in costs:
                continue
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            paths[goal] = list(reversed(path))
        return paths

    def _delivery_first_safe(self, state, path):
        if len(path) < 2:
            return False
        nxt = path[1]
        if self._home(nxt) or nxt in self.get_capsules(state):
            return True
        for enemy in self.enemies:
            status = state.get_agent_state(enemy)
            pos = self._cell(state.get_agent_position(enemy))
            if (pos is not None and not status.is_pacman and status.scared_timer <= 1
                    and self._maze(nxt, pos) <= 1):
                return False
        return True

    @staticmethod
    def _delivery_limit(remaining, baseline_eta):
        # Current callback is one of these remaining own moves.
        return min(remaining, baseline_eta + min(3, max(0, (remaining - baseline_eta) // 4)))

    @staticmethod
    def _delivery_next(pos, action):
        dx, dy = DIRECTION_STEP[action]
        return pos[0] + dx, pos[1] + dy

    def _return_route(self, pos, threats, carrying):
        normal = super()._return_route(pos, threats, carrying)
        state = getattr(self, '_tw_state', None)
        if state is None or self._home(pos) or not carrying:
            self._delivery_plan = None
            return normal
        remaining = (max(0, state.data.timeleft) + 3) // 4
        important = carrying >= 4 or remaining <= self._bank_distance(pos) + 8
        if not important:
            return normal
        tick = self._tick(state)
        plan = self._delivery_plan
        if plan and pos in plan['path']:
            path = plan['path'][plan['path'].index(pos):]
            valid_capsule = plan['capsule'] is None or plan['capsule'] in self.get_capsules(state) or plan['capsule'] not in path
            if (len(path) > 1 and len(path) - 1 <= remaining and valid_capsule
                    and self._delivery_first_safe(state, path)
                    and tick - plan['progress_tick'] <= 12):
                if len(path) < len(plan['path']):
                    plan['progress_tick'] = tick
                plan['path'] = path
                self._return_portal = path[-1]
                self._event('delivery_commit_kept')
                return self._tw_action(pos, path[1]), path[-1], 'return'
            self._event('delivery_commit_replanned')
        baseline_risk = _HistoryBase._danger_map(self, threats, carrying)
        risk = self._danger_map(threats, carrying)
        capsules = set(map(self._cell, self.get_capsules(state)))
        for capsule in capsules:
            baseline_risk[capsule] = 0
        baseline = self._delivery_paths(pos, baseline_risk)
        learned = self._delivery_paths(pos, risk)
        direct = self._delivery_paths(pos, {})
        all_paths = list(baseline.values()) + list(learned.values()) + list(direct.values())
        # Include short powered exits, without assuming power before consumption.
        for capsule in sorted(capsules):
            prefix, _ = self._tw_path(pos, [capsule], risk)
            if prefix:
                tail, _ = self._tw_path(capsule, self.home_portals, {})
                if tail and len(tail) - 1 <= 30:
                    all_paths.append(prefix + tail[1:])
        safe_baseline = [p for p in baseline.values() if self._delivery_first_safe(state, p)]
        reference = min(safe_baseline or list(baseline.values()),
                        key=lambda p: sum(1 + baseline_risk.get(c, 0) for c in p[1:]), default=None)
        if reference is None:
            return normal
        limit = self._delivery_limit(remaining, len(reference) - 1)
        candidates = [p for p in all_paths if 1 < len(p) <= limit + 1
                      and self._delivery_first_safe(state, p)]
        if not candidates:
            self._delivery_plan = None
            self._event('delivery_no_feasible_safe_path')
            feasible = [p for p in all_paths if 1 < len(p) <= remaining + 1]
            if not feasible:
                self._event('delivery_deadline_unreachable')
                return normal
            candidates = feasible  # Native tactical layer still checks immediate danger.
        def cost(path):
            powered, total = 0, 0.0
            for cell in path[1:]:
                powered = 40 if cell in capsules else max(0, powered - 1)
                total += 1 + (0 if powered else risk.get(cell, 0))
            # Keep an already selected exit when its cost is close.
            return total + (0 if path[-1] == self._return_portal else .5), len(path), path
        path = min(candidates, key=cost)
        capsule = next((p for p in path[1:] if p in capsules), None)
        self._delivery_plan = {'path': path, 'progress_tick': tick, 'capsule': capsule}
        self._return_portal = path[-1]
        self._event('delivery_commit_started')
        if len(reference) - 1 > remaining:
            self._event('delivery_deadline_shortcut')
        return self._tw_action(pos, path[1]), path[-1], 'return'

    def _cooperative_capsule_rescue(self, state, pos, legal, threats, carrying, mate, normal, max_distance):
        saved = (getattr(self, '_support_resume_bank', False), getattr(self, '_support_return_capsule', None))
        choice = super()._cooperative_capsule_rescue(state, pos, legal, threats, carrying, mate, normal, max_distance)
        if not choice:
            return choice
        intent = self._team_memory['_delivery92']['intents'].get(self.other)
        if (not intent or self._tick(state) - intent['tick'] > 8
                or intent['capsule'] != choice[1] or mate is None
                or self._maze(mate, choice[1]) + 2 > self._maze(pos, choice[1])):
            return choice
        # A promise cannot protect a threatened carrier: yield only a safe normal step.
        nxt = self._delivery_next(pos, normal[0])
        if self._delivery_first_safe(state, [pos, nxt]):
            self._support_resume_bank, self._support_return_capsule = saved
            self._event('delivery_capsule_duplicate_avoided')
            return normal
        return choice

    def choose_action(self, state):
        action = super().choose_action(state)
        pos = self._cell(state.get_agent_position(self.index))
        if not state.get_agent_state(self.index).num_carrying:
            self._delivery_plan = None
        capsule = self.goal if self.goal in self.get_capsules(state) else None
        if self._delivery_plan and self.goal_kind == 'return':
            capsule = self._delivery_plan['capsule']
            path = self._delivery_plan['path']
            if len(path) < 2 or self._delivery_next(pos, action) != path[1]:
                self._delivery_plan = None
                self._event('delivery_tactical_override')
        self._team_memory['_delivery92']['intents'][self.index] = {
            'tick': self._tick(state), 'capsule': capsule,
            'goal': self.goal, 'kind': self.goal_kind}
        return action
