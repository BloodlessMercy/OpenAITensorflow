#! /usr/bin/env python3

"""Flappy birb, implemented using Pygame."""

import math
from random import randint
from collections import deque
import numpy as np
import pprint

import tensorflow as tf
from tensorflow import layers as tfl

import pygame
from pygame import Rect, QUIT, KEYUP, K_ESCAPE, K_PAUSE, K_p


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

birbS_PER_POP = 15
MATING_SIZE = 2


global pprinter
FPS = 60
ANIMATION_SPEED = 0.18  # pixels per millisecond
WIN_WIDTH = 284 * 2     # BG image size: 284x512 px; tiled twice
WIN_HEIGHT = 512
MUTATION_RATE = 0.02


class birb(pygame.sprite.Sprite):
    """Represents the birb controlled by the player.

    The birb is the 'hero' of this game.  The player can make it climb
    (ascend quickly), otherwise it sinks (descends more slowly).  It must
    pass through the space in between pipes (for every pipe passed, one
    point is scored); if it crashes into a pipe, the game ends.

    Attributes:
    x: The birb's X coordinate.
    y: The birb's Y coordinate.
    msec_to_climb: The number of milliseconds left to climb, where a
        complete climb lasts birb.CLIMB_DURATION milliseconds.

    Constants:
    WIDTH: The width, in pixels, of the birb's image.
    HEIGHT: The height, in pixels, of the birb's image.
    SINK_SPEED: With which speed, in pixels per millisecond, the birb
        descends in one second while not climbing.
    CLIMB_SPEED: With which speed, in pixels per millisecond, the birb
        ascends in one second while climbing, on average.  See also the
        birb.update docstring.
    CLIMB_DURATION: The number of milliseconds it takes the birb to
        execute a complete climb.
    """

    WIDTH = HEIGHT = 32
    SINK_SPEED = 0.18
    CLIMB_SPEED = 0.3
    CLIMB_DURATION = 333.3

    def __init__(self, x, y, msec_to_climb, images, index, neural_network):
        """Initialise a new birb instance.

        Arguments:
        x: The birb's initial X coordinate.
        y: The birb's initial Y coordinate.
        msec_to_climb: The number of milliseconds left to climb, where a
            complete climb lasts birb.CLIMB_DURATION milliseconds.  Use
            this if you want the birb to make a (small?) climb at the
            very beginning of the game.
        images: A tuple containing the images used by this birb.  It
            must contain the following images, in the following order:
                0. image of the birb with its wing pointing upward
                1. image of the birb with its wing pointing downward
        """
        super(birb, self).__init__()
        self.x, self.y = x, y
        self.index = index
        self.msec_to_climb = msec_to_climb
        self._img_wingup, self._img_wingdown = images
        self._mask_wingup = pygame.mask.from_surface(self._img_wingup)
        self._mask_wingdown = pygame.mask.from_surface(self._img_wingdown)
        self.network = neural_network
        self.score = 0
        self.fitness = 0

    def update(self, delta_frames=1):
        """Update the birb's position.

        This function uses the cosine function to achieve a smooth climb:
        In the first and last few frames, the birb climbs very little, in the
        middle of the climb, it climbs a lot.
        One complete climb lasts CLIMB_DURATION milliseconds, during which
        the birb ascends with an average speed of CLIMB_SPEED px/ms.
        This birb's msec_to_climb attribute will automatically be
        decreased accordingly if it was > 0 when this method was called.

        Arguments:
        delta_frames: The number of frames elapsed since this method was
            last called.
        """
        if self.msec_to_climb > 0:
            frac_climb_done = 1 - self.msec_to_climb/birb.CLIMB_DURATION
            self.y -= (birb.CLIMB_SPEED * frames_to_msec(delta_frames) *
                       (1 - math.cos(frac_climb_done * math.pi)))
            self.msec_to_climb -= frames_to_msec(delta_frames)
        else:
            self.y += birb.SINK_SPEED * frames_to_msec(delta_frames)

    @property
    def image(self):
        """Get a Surface containing this birb's image.

        This will decide whether to return an image where the birb's
        visible wing is pointing upward or where it is pointing downward
        based on pygame.time.get_ticks().  This will animate the flapping
        birb, even though pygame doesn't support animated GIFs.
        """
        if pygame.time.get_ticks() % 500 >= 250:
            return self._img_wingup
        else:
            return self._img_wingdown

    @property
    def mask(self):
        """Get a bitmask for use in collision detection.

        The bitmask excludes all pixels in self.image with a
        transparency greater than 127."""
        if pygame.time.get_ticks() % 500 >= 250:
            return self._mask_wingup
        else:
            return self._mask_wingdown

    @property
    def rect(self):
        """Get the birb's position, width, and height, as a pygame.Rect."""
        return Rect(self.x, self.y, birb.WIDTH, birb.HEIGHT)


class PipePair(pygame.sprite.Sprite):
    """Represents an obstacle.

    A PipePair has a top and a bottom pipe, and only between them can
    the birb pass -- if it collides with either part, the game is over.

    Attributes:
    x: The PipePair's X position.  This is a float, to make movement
        smoother.  Note that there is no y attribute, as it will only
        ever be 0.
    image: A pygame.Surface which can be blitted to the display surface
        to display the PipePair.
    mask: A bitmask which excludes all pixels in self.image with a
        transparency greater than 127.  This can be used for collision
        detection.
    top_pieces: The number of pieces, including the end piece, in the
        top pipe.
    bottom_pieces: The number of pieces, including the end piece, in
        the bottom pipe.

    Constants:
    WIDTH: The width, in pixels, of a pipe piece.  Because a pipe is
        only one piece wide, this is also the width of a PipePair's
        image.
    PIECE_HEIGHT: The height, in pixels, of a pipe piece.
    ADD_INTERVAL: The interval, in milliseconds, in between adding new
        pipes.
    """

    WIDTH = 80
    PIECE_HEIGHT = 32
    ADD_INTERVAL = 3000

    def __init__(self, pipe_end_img, pipe_body_img):
        """Initialises a new random PipePair.

        The new PipePair will automatically be assigned an x attribute of
        float(WIN_WIDTH - 1).

        Arguments:
        pipe_end_img: The image to use to represent a pipe's end piece.
        pipe_body_img: The image to use to represent one horizontal slice
            of a pipe's body.
        """
        pygame.sprite.Sprite.__init__(self)
        self.x = float(WIN_WIDTH - 1)
        self.score_counted = False

        self.image = pygame.Surface((PipePair.WIDTH, WIN_HEIGHT))
        self.image.convert()   # speeds up blitting
        self.image.fill((0, 0, 0, 0))
        total_pipe_body_pieces = int(
            (WIN_HEIGHT -                  # fill window from top to bottom
             3 * birb.HEIGHT -             # make room for birb to fit through
             3 * PipePair.PIECE_HEIGHT) /  # 2 end pieces + 1 body piece
            PipePair.PIECE_HEIGHT          # to get number of pipe pieces
        )
        self.bottom_pieces = randint(1, total_pipe_body_pieces)
        self.top_pieces = total_pipe_body_pieces - self.bottom_pieces

        # bottom pipe
        for i in range(1, self.bottom_pieces + 1):
            piece_pos = (0, WIN_HEIGHT - i*PipePair.PIECE_HEIGHT)
            self.image.blit(pipe_body_img, piece_pos)
        bottom_pipe_end_y = WIN_HEIGHT - self.bottom_height_px
        bottom_end_piece_pos = (0, bottom_pipe_end_y - PipePair.PIECE_HEIGHT)
        self.image.blit(pipe_end_img, bottom_end_piece_pos)

        # top pipe
        for i in range(self.top_pieces):
            self.image.blit(pipe_body_img, (0, i * PipePair.PIECE_HEIGHT))
        top_pipe_end_y = self.top_height_px
        self.image.blit(pipe_end_img, (0, top_pipe_end_y))

        # compensate for added end pieces
        self.top_pieces += 1
        self.bottom_pieces += 1

        # for collision detection
        self.mask = pygame.mask.from_surface(self.image)

    @property
    def top_height_px(self):
        """Get the top pipe's height, in pixels."""
        return self.top_pieces * PipePair.PIECE_HEIGHT

    @property
    def bottom_height_px(self):
        """Get the bottom pipe's height, in pixels."""
        return self.bottom_pieces * PipePair.PIECE_HEIGHT

    @property
    def visible(self):
        """Get whether this PipePair on screen, visible to the player."""
        return -PipePair.WIDTH < self.x < WIN_WIDTH

    @property
    def rect(self):
        """Get the Rect which contains this PipePair."""
        return Rect(self.x, 0, PipePair.WIDTH, PipePair.PIECE_HEIGHT)

    def update(self, delta_frames=1):
        """Update the PipePair's position.

        Arguments:
        delta_frames: The number of frames elapsed since this method was
            last called.
        """
        self.x -= ANIMATION_SPEED * frames_to_msec(delta_frames)

    def birb_pipe_collision_x(self, birb):
        return self.x - (0.5 * PipePair.WIDTH) < birb.x < self.x + (0.5 * PipePair.WIDTH)

    def collides_with(self, birb):
        """Get whether the birb collides with a pipe in this PipePair.

        Arguments:
        birb: The birb which should be tested for collision with this
            PipePair.
        """
        collides = False
        if self.birb_pipe_collision_x(birb) and (birb.y < self.bottom_height_px or birb.y > self.top_height_px):
            collides = True
        return collides


class NeuralNetwork(tf.keras.Sequential):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        model = tf.keras.Sequential([tfl.Dense(4, activation=tf.nn.relu, input_shape=[4]),
                                    tfl.Dense(8, activation=tf.nn.relu),
                                    tfl.Dense(1, activation=tf.nn.relu)])
        adam = tf.keras.optimizers.Adam()
        model.compile(loss='mse',
                      optimizer=adam,
                      metrics=['mae', 'mse'])
        self.model = model


def load_images():
    """Load all images required by the game and return a dict of them.

    The returned dict has the following keys:
    background: The game's background image.
    birb-wingup: An image of the birb with its wing pointing upward.
        Use this and birb-wingdown to create a flapping birb.
    birb-wingdown: An image of the birb with its wing pointing downward.
        Use this and birb-wingup to create a flapping birb.
    pipe-end: An image of a pipe's end piece (the slightly wider bit).
        Use this and pipe-body to make pipes.
    pipe-body: An image of a slice of a pipe's body.  Use this and
        pipe-body to make pipes.
    """

    def load_image(img_file_name):
        """Return the loaded pygame image with the specified file name.

        This function looks for images in the game's images folder
        (./images/).  All images are converted before being returned to
        speed up blitting.

        Arguments:
        img_file_name: The file name (including its extension, e.g.
            '.png') of the required image, without a file path.
        """
        file_name = os.path.join('.', 'images', img_file_name)
        img = pygame.image.load(file_name)
        img.convert()
        return img

    return {'background': load_image('background.png'),
            'pipe-end': load_image('pipe_end.png'),
            'pipe-body': load_image('pipe_body.png'),
            # images for animating the flapping birb -- animated GIFs are
            # not supported in pygame
            'birb-wingup': load_image('birb_wing_up.png'),
            'birb-wingdown': load_image('birb_wing_down.png')}


def frames_to_msec(frames, fps=FPS):
    """Convert frames to milliseconds at the specified framerate.

    Arguments:
    frames: How many frames to convert to milliseconds.
    fps: The framerate to use for conversion.  Default: FPS.
    """
    return 1000.0 * frames / fps


def msec_to_frames(milliseconds, fps=FPS):
    """Convert milliseconds to frames at the specified framerate.

    Arguments:
    milliseconds: How many milliseconds to convert to frames.
    fps: The framerate to use for conversion.  Default: FPS.
    """
    return fps * milliseconds / 1000.0


def calculate_fitness(birbs_with_scores):
    sum_scores = 0
    for birb in birbs_with_scores.items():
        sum_scores += birb[1].score
    for birb in birbs_with_scores.items():
        birb[1].fitness = birb[1].score / sum_scores
    return birbs_with_scores


def pool_selection(birbs):
    index = 0
    r = np.random.uniform(0, 1)
    # Keep subtracting probabilities until you get less than zero
    # Higher probabilities will be more likely to be fixed since they will
    # subtract a larger number towards zero
    while r > 0:
        if index in range(len(birbs)):
            r -= birbs[index].fitness
            # And move on to the next
            index += 1

    # Go back one
    index -= 1
    # Make sure it's a copy!
    # (this includes mutation)
    birbs[index].fitness = 0
    return birbs[index]


def select_parents(number_of_parents, birbs_with_fitness):
    parent_birbs = []
    for _ in range(number_of_parents):
        parent_birbs.append(pool_selection(birbs_with_fitness))
    return parent_birbs


def main(initial_nn_pop, max_generations=10):
    """The application's entry point.

    If someone executes this module (instead of importing it, for
    example), this function is called.
    """
    generation = 0
    global display_surface
    display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    global images
    images = load_images()
    pygame.init()
    birbs = create_birbs(birbS_PER_POP, images, initial_nn_pop)
    while generation < max_generations:
        birbs_with_scores = run_game(birbs)
        updated_birbs = generate_new_generation(birbs_with_scores)
        birbs = pygame.sprite.Group()
        for index, birb in updated_birbs.items():
            birb.y = (0.5 * WIN_HEIGHT) - (0.5 * birb.HEIGHT)
            birb.msec_to_climb = 0
            birbs.add(birb)
        generation += 1
    pygame.quit()


def create_child_brother(updated_birb: birb, index: int):
    model_weights = np.array(updated_birb.network.get_weights())
    brother_birb = birb(50, int(WIN_HEIGHT / 2 - birb.HEIGHT / 2),
                        2, (images['birb-wingup'], images['birb-wingdown']), index, NeuralNetwork())
    for layer_idx, layer in enumerate(model_weights):
        for l_w_idx, layer_weight in enumerate(layer):
            if layer_weight is not 0 and hasattr(layer_weight, '__iter__'):
                for w_idx, weight in enumerate(layer_weight):
                    model_weights[layer_idx][l_w_idx][w_idx] = np.random.normal(weight, abs(0.25 * weight))
    brother_birb.network.set_weights(model_weights)
    return brother_birb


def create_offspring_network(birb, selected_parent_birbs):
    updated_birb = create_crossover_mutated_child(birb, selected_parent_birbs)
    selected_parent_birbs.append(birb)
    offspring = selected_parent_birbs
    index = 1 + MATING_SIZE
    while len(offspring) < birbS_PER_POP:
        offspring.append(create_child_brother(updated_birb, index))
        index += 1
    return offspring


def create_crossover_mutated_child(birb: birb, selected_parent_birbs):
    model_weights = np.array(birb.network.get_weights())
    parent_a_weights = np.array(selected_parent_birbs[0].network.get_weights())
    parent_b_weights = np.array(selected_parent_birbs[1].network.get_weights())
    parent_a_range = (1.0 - MUTATION_RATE) / 2.0
    parent_b_range = 1.0 - MUTATION_RATE
    for layer_idx, layer in enumerate(model_weights):
        for l_w_idx, layer_weight in enumerate(layer):
            if layer_weight is not 0 and hasattr(layer_weight, '__iter__'):
                for w_idx, _ in enumerate(layer_weight):
                    random = np.random.uniform(0, 1)
                    if random > parent_b_range:
                        continue
                    if random <= parent_a_range:
                        model_weights[layer_idx][l_w_idx][w_idx] = parent_a_weights[layer_idx][l_w_idx][w_idx]
                    if parent_b_range > random > parent_a_range:
                        model_weights[layer_idx][l_w_idx][w_idx] = parent_b_weights[layer_idx][l_w_idx][w_idx]

    birb.network.set_weights(model_weights)
    return birb


def generate_new_generation(birbs_with_scores):
    new_generation = create_birbs(birbS_PER_POP, images, initial_nn_pop)
    birbs_with_fitness = calculate_fitness(birbs_with_scores)
    selected_parent_birbs = select_parents(MATING_SIZE, birbs_with_fitness)
    for birb in new_generation:
         birb = create_offspring_network(birb, selected_parent_birbs)
    return birbs_with_scores


def calculate_distance_to_next_pipe(closest_pipe: PipePair):
    if closest_pipe is not None:
        return (closest_pipe.x - 50) / WIN_WIDTH
    else:
        return 1.0


def find_next_pipe_pair(pipes):
    closest_distance = float('+Inf')
    closest_pipe = None
    for pipe in pipes:
        distance = pipe.x - 50
        if 0 < distance < closest_distance:
            closest_distance = distance
            closest_pipe = pipe
    return closest_pipe


def calculate_next_pipe_opening_height(closest_pipe: PipePair):
    if closest_pipe is not None:
        return (closest_pipe.bottom_pieces * PipePair.PIECE_HEIGHT) / WIN_HEIGHT
    else:
        return 0.5


def run_game(birbs):
    score_list = {}
    pygame.display.set_caption('Pygame Flappy birb')
    clock = pygame.time.Clock()
    score_font = pygame.font.SysFont(None, 32, bold=True)  # default font
    pipes = deque()
    frame_clock = 0  # this counter is only incremented if the game isn't paused
    score = 0
    done = paused = False
    while not done:
        clock.tick(FPS)

        # check for collisions
        for birb in birbs.sprites():
            for p in pipes:
                pipe_collision = p.collides_with(birb)
                if pipe_collision or 0 >= birb.y or birb.y >= WIN_HEIGHT - birb.HEIGHT:
                    birb.score = score
                    score_list[birb.index] = birb
                    birbs.remove(birb)
        # Handle this 'manually'.  If we used pygame.time.set_timer(),
        # pipe addition would be messed up when paused.
        if not (paused or frame_clock % msec_to_frames(PipePair.ADD_INTERVAL)):
            pp = PipePair(images['pipe-end'], images['pipe-body'])
            pipes.append(pp)

        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                done = True
                break
            elif e.type == KEYUP and e.key in (K_PAUSE, K_p):
                paused = not paused

        next_pipe_pair = find_next_pipe_pair(pipes)
        relative_distance_to_next_pipe = calculate_distance_to_next_pipe(next_pipe_pair)
        relative_height_of_pipe_opening = calculate_next_pipe_opening_height(next_pipe_pair)

        for birb in birbs.sprites():
            relative_height = birb.y / WIN_HEIGHT
            relative_falling_time = birb.msec_to_climb / birb.CLIMB_DURATION
            predict_input = np.array([
                [
                    relative_height,
                    relative_distance_to_next_pipe,
                    relative_falling_time,
                    relative_height_of_pipe_opening
                ]
            ])
            jump = birb.network.predict(predict_input)
            if jump[0][0] > 0.5 and birb.msec_to_climb < 20:
                birb.msec_to_climb = birb.CLIMB_DURATION

        if paused:
            continue  # don't draw anything

        if len(birbs.sprites()):
            for x in (0, WIN_WIDTH / 2):
                display_surface.blit(images['background'], (x, 0))

        while pipes and not pipes[0].visible:
            pipes.popleft()

        for p in pipes:
            p.update()
            display_surface.blit(p.image, p.rect)
        score += 1
        birbs.update()
        birbs.draw(display_surface)
        for birb in birbs:
            display_surface.blit(birb.image, birb.rect)
        score_surface = score_font.render(str(score), True, (255, 255, 255))
        score_x = WIN_WIDTH / 2 - score_surface.get_width() / 2
        display_surface.blit(score_surface, (score_x, PipePair.PIECE_HEIGHT))
        if not len(birbs.spritedict):
            done = True
        pygame.display.flip()
        frame_clock += 1
    print("Game over! Scores: \n\t")
    for index, birb in score_list.items():
        print(index, ' - ', birb.score)

    return score_list


def create_birbs(number, images, networks):
    birbs = pygame.sprite.Group()
    for i in range(number):
        birbs.add(birb(50, int(WIN_HEIGHT / 2 - birb.HEIGHT / 2),
                       2, (images['birb-wingup'], images['birb-wingdown']), i, networks[i]))
    return birbs


if __name__ == '__main__':
    # If this module had been imported, __name__ would be 'flappybirb'.
    # It was executed (e.g. by double-clicking the file), so call main.
    pprinter = pprint.PrettyPrinter(indent=4)
    initial_nn_pop = []
    for _ in range(birbS_PER_POP):
        initial_nn_pop.append(NeuralNetwork())
    main(initial_nn_pop)
