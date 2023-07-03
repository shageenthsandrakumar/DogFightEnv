import gymnasium as gym
import numpy as np
import pygame

from typing import Optional, Union
#import GameEntities

class DogFightEnv(gym.Env):
    metadata = {
        "render_fps" : 10,
        "render_modes" : ["human"]
    }

    def __init__(self, render_mode = None):
        # Episode time & time limit
        self.step_count = 0
        self.step_limit = 45 * self.metadata["render_fps"]
        self.tau = 1 / self.metadata["render_fps"]

        # Rendering settings
        self.render_mode = render_mode
        self.screen_size = (900, 900)
        self.screen = None
        self.clock = None
        self.min_x = 0
        self.max_x = self.screen_size[0]
        self.min_y = 0
        self.max_y = self.screen_size[1]

        # Game area settings
        self.origin = (0.5 * self.screen_size[0], 0.5 * self.screen_size[1])
        self.world_area = (0, 0, self.screen_size[0], self.screen_size[1])
        self.game_area = (100, 100, 700, 700)

        # Jet properties
        self.STRAIGHT =  0
        self.LEFT     =  1
        self.RIGHT    = -1

        # Player properties
        self.player_radius = 10
        self.player_min_speed = 7.0 / self.tau
        self.player_max_speed = 7.0 / self.tau
        self.player_min_acceleration = -(self.player_max_speed - self.player_min_speed) / 0.25
        self.player_max_acceleration =  (self.player_max_speed - self.player_min_speed) / 2.50
        self.player_min_turn_rate = np.pi / 4.0
        self.player_max_turn_rate = np.pi / 2.0
        self.player_observation_range = 0.25 * (self.screen_size[0] - 100)

        # Player missile properties
        self.player_missile_id_counter = 0
        self.player_missile_radius = 3
        self.player_missile_speed = 1.25 * self.player_max_speed
        self.player_missile_range = 3 * self.player_observation_range
        self.player_missile_angle_offset = 0.04

        # Enemy properties
        self.enemy_radius = 10
        self.enemy_min_speed = 0.75 * self.player_min_speed
        self.enemy_max_speed = 0.75 * self.player_max_speed
        self.enemy_min_acceleration = -(self.enemy_max_speed - self.enemy_min_speed) / 0.25
        self.enemy_max_acceleration =  (self.enemy_max_speed - self.enemy_min_speed) / 2.50
        self.enemy_min_turn_rate = np.pi / 4.0
        self.enemy_max_turn_rate = np.pi / 2.0
        self.enemy_observation_range = 0.75 * 0.25 * (self.screen_size[0] - 100)
        self.enemy_accel_fov = 0.75
        self.enemy_attack_fov = 0.35
        max_turn_time = np.pi / (self.enemy_min_turn_rate)
        turn_margin = max_turn_time * self.enemy_max_speed
        self.enemy_turn_area = (
            turn_margin,
            turn_margin,
            (self.screen_size[0] - 100) - 2 * turn_margin,
            (self.screen_size[0] - 100) - 2 * turn_margin
        )

        # Enemy missile properties
        self.enemy_missile_radius = 3
        self.enemy_missile_speed = 1.25 * self.enemy_max_speed
        self.enemy_missile_range = 3 * self.player_observation_range
        self.enemy_missile_angle_offset = 0.04

        # Target properties
        self.target_radius = 3.0 * self.player_radius
        self.target_opening_angle = 0.35 * np.pi
        self.target_opening_range = 0.50 * self.player_missile_range

        # Reward definitions
        self.reward_missile_miss = 0
        self.reward_missile_hit_enemy = 50
        self.reward_missile_hit_target = 100
        self.reward_player_collides_with_enemy = -500
        self.reward_player_out_of_bounds = -2
        self.reward_player_leaves_game = -100
        self.reward_time_penalty = -1 / self.metadata["render_fps"]
        self.reward_approach_target = abs(self.reward_time_penalty)

        # Environment observation space:
        #  0.) Jet absolute x position
        #  1.) Jet absolute y position
        #  2.) Jet absolute angle
        #  3.) Target absolute x position
        #  4.) Target absolute y position
        #  5.) Target absolute angle
        #  6.) Target distance
        #  7.) Enemy absolute x position
        #  8.) Enemy absolute y position
        #  9.) Enemy absolute angle
        # 10.) Enemy distance
        self.observation_space = gym.spaces.Box(
            low = np.array([
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
            ]),
            high = np.array([
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
            ]),
            dtype = np.float64
        )

        # Environment action space:
        # 0.) Do nothing (NO-OP)
        # 1.) Fire missile at target
        # 2.) Fire missile at enemy
        # 3.) Turn left
        # 4.) Turn right
        self.action_space = gym.spaces.Discrete(5)

    def step(self, action):
        # Additional dictionary to track missile information for assigning
        # delayed rewards to the proper step (done outside the environment)
        player_missile_step_info = {
            "shoot_id"      : None, # the id of the missile shot during this step
            "hit_ids"      : [],   # the ids of the missiles that hit this step
            "miss_ids"     : [],   # the ids of the missiles that missed this step
            "hit_rewards"  : [],   # rewards for the missiles that hit
            "miss_rewards" : [],   # rewards (penalties) for the missiles that missed
        }

        # Handle the player actions
        player_turn_direction = self.STRAIGHT
        # The player shoots at the target or the enemy
        if action == 1:
            if (
                self.target.distance_to(self.player) < self.target_opening_range and
                self.target.in_fov(self.player)
            ):
                fired_in_zone = True
            else:
                fired_in_zone = False
            new_missile = Missile(
                self.player.x,
                self.player.y,
                speed = self.player_missile_speed,
                angle = self.player.angle + self.player.angle_to(self.target),
                radius = self.player_missile_radius,
                range = self.player_missile_range,
                id = self.player_missile_id_counter,
                fired_in_zone = fired_in_zone
            )
            self.player.missiles.append(new_missile)
            player_missile_step_info["shoot_id"] = self.player_missile_id_counter
            self.player_missile_id_counter += 1
        # The player shoots at the enemy
        elif action == 2:
            if self.enemy is not None and self.player.distance_to(self.enemy) <= self.player_observation_range:
#                pos = (
#                    self.enemy.x + 0.25 * self.enemy.speed * np.cos(self.enemy.angle),
#                    self.enemy.x + 0.25 * self.enemy.speed * np.sin(self.enemy.angle)
#                )
                new_missile = Missile(
                    self.player.x,
                    self.player.y,
                    speed = self.player_missile_speed,
                    angle = self.player.angle + self.player.angle_to(self.enemy),
                    radius = self.player_missile_radius,
                    range = self.player_missile_range,
                    id = self.player_missile_id_counter
                )
                self.player.missiles.append(new_missile)
                player_missile_step_info["shoot_id"] = self.player_missile_id_counter
                self.player_missile_id_counter += 1
        # The player turns left or right
        elif action == 3 or action == 4:
            player_turn_direction = [self.LEFT, self.RIGHT][action - 3]

        # Handle the enemy actions (if it is still alive)
        if self.enemy is not None:
            enemy_turn_direction = self.STRAIGHT
            distance_to_player = self.enemy.distance_to(self.player)
            player_in_range = (distance_to_player <= self.enemy.observation_range)

            # If the player is not in range
            if not player_in_range:
                # Does the enemy have a guess as to where the player went?
                if self.enemy.guess_position is not None:
                    # Approach the position where the enemy predicts the player to be
                    if self.enemy.distance_to(self.enemy.guess_position) > 0.5 * self.enemy.observation_range:
                        enemy_turn_direction = self.enemy.get_turn_direction_to(self.enemy.guess_position)
                    # If the enemy arrives to the predicted position and finds nothing, stop pursuing
                    else:
                        self.enemy.guess_position = None
                # Turn back to the center of the game area if the enemy has no target to pursue
                elif not self.enemy.in_area(*(self.enemy_turn_area)):
                    enemy_turn_direction = self.enemy.get_turn_direction_to(self.origin)
            # If the player is in range
            else:
                # Chase the player
                enemy_turn_direction = self.enemy.get_turn_direction_to(self.player)
                # Constantly predict where the player will go in case the enemy loses vision
                self.enemy.guess_position = (
                    self.player.x + 2.5 * self.player.speed * np.cos(self.player.angle),
                    self.player.y + 2.5 * self.player.speed * np.sin(self.player.angle)
                )

        # Move the player
        p_oob = False
        pold_x, pold_y, pold_a = self.player.x, self.player.y, self.player.angle
        self.player.move(self.tau, move_direction = player_turn_direction)
        if not self.player.in_area(*self.world_area):
            self.player.x, self.player.y, self.player.angle = pold_x, pold_y, pold_a
            p_oob = True

        # Move the enemy
        if self.enemy is not None:
            self.enemy.move(self.tau, turn_direction = enemy_turn_direction)

        # ================================================================================
        # Player Missile Movement & Delayed Reward Handling
        # ================================================================================
        terminated = False
        for missile in self.player.missiles[:]:
            missile.move(self.tau)
            # Missles that go beyond their range or out of bounds are removed and
            # count as a miss
            if not missile.in_area(*(self.world_area)):
                player_missile_step_info["miss_ids"].append(missile.id)
                player_missile_step_info["miss_rewards"].append(self.reward_missile_miss)
                self.player.missiles.remove(missile)
            elif missile.range < missile.distance_to(missile.origin):
                player_missile_step_info["miss_ids"].append(missile.id)
                player_missile_step_info["miss_rewards"].append(self.reward_missile_miss)
                self.player.missiles.remove(missile)
            elif self.enemy is not None and missile.collides_with(self.enemy):
                player_missile_step_info["hit_ids"].append(missile.id)
                player_missile_step_info["hit_rewards"].append(self.reward_missile_hit_enemy)
                self.enemy = None
                self.player.missiles.remove(missile)
            elif missile.collides_with(self.target) and missile.fired_in_zone == True:
                player_missile_step_info["hit_ids"].append(missile.id)
                player_missile_step_info["hit_rewards"].append(self.reward_missile_hit_target)
                self.player.missiles.remove(missile)
                terminated = True
                break
        # ================================================================================

        # ================================================================================
        # Immediate Rewards
        # ================================================================================
        reward, truncated = 0, False

        # The player collides with the enemy
        if self.enemy is not None and self.enemy.collides_with(self.player):
            reward += self.reward_player_collides_with_enemy
            terminated = True
        else:
            # Constant negative reward to encourage the agent to finish sooner
            reward += self.reward_time_penalty
            if p_oob:
                reward += self.reward_player_leaves_game
            distance_target_to_player = self.player.distance_to(self.target)
            if (
                distance_target_to_player <= self.target_opening_range and
                self.target.in_fov(self.player)
            ):
                reward += self.reward_approach_target
            if self.enemy is not None:
                distance_to_player = self.enemy.distance_to(self.player)
                if distance_to_player <= 0.50 * self.enemy.observation_range:
                    reward += 8 * self.reward_time_penalty
                elif distance_to_player <= self.enemy_observation_range:
                    reward += 4 * self.reward_time_penalty
        # ================================================================================

        if self.render_mode == "human":
            self.render()

        px_norm = self.player.x / self.max_x
        py_norm = self.player.y / self.max_y
        pa_norm = self.player.angle / np.pi
        if (
            self.enemy is None or
            (self.player.distance_to(self.enemy) > self.player.observation_range)
        ):
            ex_norm = -1.
            ey_norm = -1.
            ea_norm = -1.
            ed_norm = -1.
        else:
            ex_norm = self.enemy.x / self.max_x
            ey_norm = self.enemy.y / self.max_y
            ea_norm = self.enemy.angle / np.pi
            ed_norm = self.player.distance_to(self.enemy) / self.max_x
        tx_norm = self.target.x / self.max_x
        ty_norm = self.target.y / self.max_y
        ta_norm = self.target.angle / np.pi
        td_norm = self.player.distance_to(self.target) / self.max_x

        self.state = (
            px_norm,
            py_norm,
            pa_norm,
            ex_norm,
            ey_norm,
            ea_norm,
            ed_norm,
            tx_norm,
            ty_norm,
            ta_norm,
            td_norm
        )

        self.step_count += 1
        if self.step_count >= self.step_limit:
            truncated = True

        return np.array(self.state, dtype = np.float64), reward, terminated, truncated, player_missile_step_info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed = seed)
        self.step_count = 0

        self.player = Player(
            self.origin[0],
            self.origin[1],
            speed = self.player_min_speed,
            angle = 0.5 * np.pi,
            min_speed = self.player_min_speed,
            max_speed = self.player_max_speed,
            min_turn_rate = self.player_min_turn_rate,
            max_turn_rate = self.player_max_turn_rate,
            radius = self.player_radius,
            observation_range = self.player_observation_range,
        )

        exy = [self.player.x, self.player.y]
        while self.player.distance_to(exy) < self.player.observation_range:
            exy[0] = np.random.uniform(self.game_area[0], self.game_area[0] + self.game_area[2])
            exy[1] = np.random.uniform(self.game_area[1], self.game_area[1] + self.game_area[3])
        dx = exy[0] - 0.5 * self.max_x
        dy = exy[1] - 0.5 * self.max_y
        ea = (np.arctan2(dy, dx) + np.pi)
        ea = np.arctan2(np.sin(ea), np.cos(ea))
        self.enemy = Jet(
            exy[0],
            exy[1],
            angle = ea,
            speed = self.enemy_min_speed,
            min_speed = self.enemy_min_speed,
            max_speed = self.enemy_max_speed,
            min_turn_rate = self.enemy_min_turn_rate,
            max_turn_rate = self.enemy_max_turn_rate,
            radius = self.enemy_radius,
            observation_range = self.enemy_observation_range,
        )

        txy = [self.player.x, self.player.y]
        while self.player.distance_to(txy) < self.player.observation_range:
            txy[0] = np.random.uniform(self.game_area[0], self.game_area[0] + self.game_area[2])
            txy[1] = np.random.uniform(self.game_area[1], self.game_area[1] + self.game_area[3])
        dx = txy[0] - 0.5 * self.max_x
        dy = txy[1] - 0.5 * self.max_y
        ta = (np.arctan2(dy, dx) + np.pi)
        ta = np.arctan2(np.sin(ta), np.cos(ta))
        self.target = Entity(
            txy[0],
            txy[1],
            angle = ta,
            radius = self.target_radius,
            fov = self.target_opening_angle
        )

        tx_trans = self.target.x - self.player.x
        ty_trans = self.target.y - self.player.y
        tx_p = tx_trans * np.cos(self.player.angle) - ty_trans * np.sin(self.player.angle)
        ty_p = tx_trans * np.sin(self.player.angle) + ty_trans * np.cos(self.player.angle)
        tx_norm = tx_p / self.max_x
        ty_norm = ty_p / self.max_y
        self.state = (
            self.player.x / self.max_x,
            self.player.y / self.max_y,
            self.player.angle / np.pi,
            -1.,
            -1.,
            -1.,
            -1.,
            self.target.x / self.max_x,
            self.target.y / self.max_y,
            self.target.angle / np.pi,
            self.player.distance_to(self.target) / self.max_x
        )

        if self.render_mode == "human":
            self.render

        return np.array(self.state, dtype = np.float64), {}

    def render(self):
        if self.screen is None:
            if self.render_mode == "human":
                pygame.display.init()
                self.screen  = pygame.display.set_mode(self.screen_size)
                self.alpha_surface = pygame.Surface(self.screen_size, pygame.SRCALPHA)
                self.lower_surface = pygame.Surface(self.screen_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state is None:
            return None

#        self.screen.fill((0, 0, 0))
        self.alpha_surface.fill((0, 0, 0, 0))
        self.lower_surface.fill((64, 0, 0))

        # Draw the transparent stuff first
        pygame.draw.rect(self.lower_surface, (0, 0, 0, 255), pygame.Rect(*(self.game_area)))
        target_firing_zone_vertices = [
            (
                self.target.x,
                self.max_y - self.target.y
            ),
            (
                self.target.x + self.target_opening_range * np.cos(self.target.angle - self.target_opening_angle / 2),
                self.max_y - (self.target.y + self.target_opening_range * np.sin(self.target.angle - self.target_opening_angle / 2)),
            ),
            (
                self.target.x + self.target_opening_range * np.cos(self.target.angle + self.target_opening_angle / 2),
                self.max_y - (self.target.y + self.target_opening_range * np.sin(self.target.angle + self.target_opening_angle / 2))
            )
        ]
        pygame.draw.polygon(self.alpha_surface, (0, 255, 255, 64), target_firing_zone_vertices)
        pygame.draw.circle(
            self.alpha_surface,
            (0, 255, 0, 64),
            (self.player.x, self.max_y - self.player.y),
            self.player.observation_range
        )
        if self.enemy is not None:
            pygame.draw.circle(
                self.alpha_surface,
                (255, 0, 0, 64),
                (self.enemy.x, self.max_y - self.enemy.y),
                self.enemy.observation_range
            )

        # Now draw the opaque stuff
#        pygame.draw.circle()
        player_vertices = [
            (
                self.player.x + self.player.radius * np.cos(self.player.angle),
                self.max_y - (self.player.y + self.player.radius * np.sin(self.player.angle))
            ),
            (
                self.player.x + (0.75 * self.player.radius) * np.cos(self.player.angle + 2*np.pi/3),
                self.max_y - (self.player.y + (0.75 * self.player.radius) * np.sin(self.player.angle + 2*np.pi/3))
            ),
            (
                self.player.x + (0.75 * self.player.radius) * np.cos(self.player.angle + 4*np.pi/3),
                self.max_y - (self.player.y + (0.75 * self.player.radius) * np.sin(self.player.angle + 4*np.pi/3))
            )
        ]
        if self.enemy is not None:
            enemy_vertices = [
                (
                    self.enemy.x + self.enemy.radius * np.cos(self.enemy.angle),
                    self.max_y - (self.enemy.y + self.enemy.radius * np.sin(self.enemy.angle))
                ),
                (
                    self.enemy.x + (0.75 * self.enemy.radius) * np.cos(self.enemy.angle + 2*np.pi/3),
                    self.max_y - (self.enemy.y + (0.75 * self.enemy.radius) * np.sin(self.enemy.angle + 2*np.pi/3))
                ),
                (
                    self.enemy.x + (0.75 * self.enemy.radius) * np.cos(self.enemy.angle + 4*np.pi/3),
                    self.max_y - (self.enemy.y + (0.75 * self.enemy.radius) * np.sin(self.enemy.angle + 4*np.pi/3))
                )
            ]
        pygame.draw.circle(self.lower_surface, (0, 0, 255, 255), (int(self.target.x), self.max_y - int(self.target.y)), int(self.target.radius))
        for missile in self.player.missiles:
            pygame.draw.circle(self.lower_surface, (255, 255, 255, 255), (int(missile.x), self.max_y - int(missile.y)), missile.radius)
            
        pygame.draw.polygon(self.lower_surface, (0, 255, 0), player_vertices)
        if self.enemy is not None:
            pygame.draw.polygon(self.lower_surface, (255, 0, 0), enemy_vertices)
        self.screen.blit(self.lower_surface, (0, 0))
        self.screen.blit(self.alpha_surface, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

class Entity():
    def __init__(self, x, y, speed = 0, angle = 0, radius = 1, fov = 0):
        self.x = x
        self.y = y

        if speed < 0:
            raise ValueError(
                "Entity speed must be >= 0"
            )
        self.speed = speed

        self.angle = np.arctan2(np.sin(angle), np.cos(angle))

        if radius <= 0.:
            raise ValueError(
                "Entity 'radius' argument must be greater than 0"
            )
        self.radius = radius

        if fov < 0:
            raise ValueError(
                "Jet fov must be >= 0"
            )
        self.fov = fov

    def in_area(self, blx, bly, width, height):
        if (
            blx < self.x and
            bly < self.y and
            self.x < blx + width and
            self.y < bly + height
        ):
            return True
        else:
            return False

    def in_fov(self, other):
        if not (
            isinstance(other, Entity) or
            (isinstance(other, (list, tuple)) and len(other) == 2)
        ):
            raise TypeError(
                "Entity 'in_fov' function must take another entity or an (x, y) coordinate pair"
            )

        return np.abs(self.angle_to(other)) < 0.5 * self.fov

    def collides_with(self, other):
        distance = self.distance_to(other)

        return distance < (self.radius + other.radius)

    def distance_to(self, other):
        if isinstance(other, Entity):
            dx = other.x - self.x
            dy = other.y - self.y
        elif isinstance(other, (list, tuple)) and len(other) == 2:
            dx = other[0] - self.x
            dy = other[1] - self.y
        else:
            raise TypeError(
                "Entity 'distance_to' function must take another entity or an (x, y) coordinate pair"
            )

        return np.sqrt(dx ** 2 + dy ** 2, dtype = np.float64)

    def move(self, tau):
        # Update position
        self.x += self.speed * np.cos(self.angle) * tau
        self.y += self.speed * np.sin(self.angle) * tau

    def angle_to(self, other):
        if isinstance(other, Entity):
            dx = other.x - self.x
            dy = other.y - self.y
        elif isinstance(other, (list, tuple)) and len(other) == 2:
            dx = other[0] - self.x
            dy = other[0] - self.y
        else:
            raise TypeError(
                "Entity 'angle_to' function must take another entity or an (x, y) coordinate pair"
            )

        abs_angle = np.arctan2(dy, dx)
        rel_angle = abs_angle - self.angle

        return np.arctan2(np.sin(rel_angle), np.cos(rel_angle), dtype = np.float64)

class Missile(Entity):
    def __init__(self, x, y, speed, angle, radius, range, fov = 0, id = None, fired_in_zone = False):
        super().__init__(x, y, speed, angle, radius, fov)        
        self.origin = (x, y)
        self.id = id
        self.fired_in_zone = fired_in_zone

        if range <= 0:
            raise ValueError(
                "Missile range must be at least > 0"
            )
        self.range = range

class Jet(Entity):
    def __init__(
        self,
        x,
        y,
        speed,
        angle,
        radius,
        min_speed,
        max_speed,
        min_turn_rate,
        max_turn_rate,
        observation_range,
        fov = 0,
    ):
        super().__init__(x, y, speed, angle, radius, fov)

        if min_speed < 0 or max_speed < 0:
            raise ValueError(
                "Jet min_speed and max_speed must be >= 0"
            )
        if not min_speed <= max_speed:
            raise ValueError(
                "Jet min_speed must be <= max_speed"
            )
        self.min_speed = min_speed
        self.max_speed = max_speed

        if min_turn_rate < 0 or max_turn_rate < 0:
            raise ValueError(
                "Jet min_turn_rate and max_turn_rate must be >= 0"
            )
        if not min_turn_rate <= max_turn_rate:
            raise ValueError(
                "Jet min_turn_rate must be <= max_turn_rate"
            )
        self.min_turn_rate = min_turn_rate
        self.max_turn_rate = max_turn_rate

        if observation_range <= radius:
            raise ValueError(
                "Jet observation_range must be at least >= radius"
            )
        self.observation_range = observation_range

        self.missile = None
        self.guess_position = None

    def get_turn_direction_to(self, other):
        if not (
            isinstance(other, Entity) or
            (isinstance(other, (list, tuple)) and len(other) == 2)
        ):
            raise TypeError(
                "Jet 'get_turn_direction_to' function must take another entity or an (x, y) coordinate pair"
            )

        da = self.angle_to(other)
        if da < 0:
            return -1
        elif 0 < da:
            return 1
        else:
            return 0

    def move(self, tau, acceleration = 0, turn_direction = 0):
        # Get the turn rate
        if turn_direction != 0:
            dv = self.max_speed - self.min_speed
            if dv == 0:
                turn_rate = self.min_turn_rate
            else:
                dw = self.max_turn_rate - self.min_turn_rate
                turn_rate = (
                    (self.max_speed - self.speed) / dv * dw + self.min_turn_rate
                )
                turn_rate = np.clip(turn_rate, self.min_turn_rate, self.max_turn_rate)
        else:
            turn_rate = 0

        # Update the angle
        self.angle += turn_direction * turn_rate * tau
        self.angle = np.arctan2(np.sin(self.angle), np.cos(self.angle))

        # Accelerate
        self.speed += acceleration * tau
        self.speed = np.clip(self.speed, self.min_speed, self.max_speed)

        # Update position
        self.x += self.speed * np.cos(self.angle) * tau
        self.y += self.speed * np.sin(self.angle) * tau

class Player(Jet):
    def __init__(
        self,
        x,
        y,
        speed,
        angle,
        radius,
        min_speed,
        max_speed,
        min_turn_rate,
        max_turn_rate,
        observation_range,
        fov = 0,
    ):
        super().__init__(
            x,
            y,
            speed,
            angle,
            radius,
            min_speed,
            max_speed,
            min_turn_rate,
            max_turn_rate,
            observation_range,
            fov
        )
        self.missiles = []

    def move(self, tau, move_direction):
        if move_direction == -1:
            self.angle -= 0.5 * np.pi
        elif move_direction == 1:
            self.angle += 0.5 * np.pi
        self.angle = np.arctan2(np.sin(self.angle), np.cos(self.angle))

        # Update position
        self.x += self.speed * np.cos(self.angle) * tau
        self.y += self.speed * np.sin(self.angle) * tau

class Enemy(Entity):
    def __init__(
        self,
        x,
        y,
        speed,
        angle,
        radius,
        min_speed,
        max_speed,
        min_turn_rate,
        max_turn_rate,
        observation_range,
        fov = 0,
    ):
        super().__init__(
            x,
            y,
            speed,
            angle,
            radius,
            min_speed,
            max_speed,
            min_turn_rate,
            max_turn_rate,
            observation_range,
            fov
        )
        self.missile = None
