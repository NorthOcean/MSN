"""
@Author: Conghao Wong
@Date: 2022-06-21 20:36:21
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-10 11:30:10
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

from ..base import BaseManager, SecondaryBar
from ..dataset.trajectories import Agent, AnnotationManager, VideoClip
from ..utils import (DISTRIBUTION_IMAGE, DRAW_TEXT_IN_IMAGES,
                     DRAW_TEXT_IN_VIDEOS, GT_IMAGE, NEIGHBOR_IMAGE, OBS_IMAGE,
                     PRED_IMAGE)
from .__helper import ADD, get_helper

DRAW_ON_VIDEO = 0
DRAW_ON_IMAGE = 1
DRAW_ON_EMPTY = 2


class Visualization(BaseManager):

    def __init__(self, manager: BaseManager,
                 dataset: str, clip: str,
                 name='Visualization Manager'):

        super().__init__(manager=manager, name=name)

        # Get information of the video clip
        self.info = VideoClip(name=clip, dataset=dataset).get()

        # Try to open the video
        path = self.info.video_path
        vc = cv2.VideoCapture(path)
        self._vc = vc if vc.open(path) else None

        # Try to read the scene image
        try:
            self.scene_image = cv2.imread(path)
        except:
            self.scene_image = None

        # annotation helper
        self.helper = get_helper(self.args.anntype)

        # Read png files
        self.obs_file = cv2.imread(OBS_IMAGE, -1)
        self.neighbor_file = cv2.imread(NEIGHBOR_IMAGE, -1)
        self.pred_file = cv2.imread(PRED_IMAGE, -1)
        self.gt_file = cv2.imread(GT_IMAGE, -1)
        self.dis_file = cv2.imread(DISTRIBUTION_IMAGE, -1)

    @property
    def video_capture(self) -> cv2.VideoCapture:
        return self._vc

    @property
    def picker(self) -> AnnotationManager:
        return self.manager.get_member(AnnotationManager)

    def get_image(self, frame: int) -> np.ndarray:
        """
        Get a frame from the video

        :param frame: The frame number of the image.
        """
        time = 1000 * frame / self.info.paras[1]
        self.video_capture.set(cv2.CAP_PROP_POS_MSEC, time - 1)
        _, f = self.video_capture.read()
        f = self.rescale(f)
        return f

    def get_text(self, frame: int, agent: Agent) -> list[str]:
        return [self.info.name,
                f'frame: {str(frame).zfill(6)}',
                f'agent: {agent.id}',
                f'type: {agent.type}']

    def get_trajectories(self, agent: Agent, integer=True):
        obs = self.real2pixel(agent.traj, integer)
        pred = self.real2pixel(agent.pred, integer)
        gt = self.real2pixel(agent.groundtruth, integer)
        nei = self.real2pixel(agent.traj_neighbor[:, -1], integer)

        if pred.ndim == 2:
            pred = pred[np.newaxis]

        return obs, pred, gt, nei

    def real2pixel(self, real_pos, integer=True):
        """
        Transfer coordinates from real scale to pixels.

        :param real_pos: Coordinates, shape = (n, 2) or (k, n, 2).
        :return pixel_pos: Coordinates in pixels.
        """
        scale = self.info.datasetInfo.scale / self.info.datasetInfo.scale_vis
        weights = self.info.matrix

        if type(real_pos) == list:
            real_pos = np.array(real_pos)

        w = [weights[0], weights[2]]
        b = [weights[1], weights[3]]

        order = self.info.order
        real = scale * real_pos
        real_2d = self.manager.get_member(AnnotationManager) \
            .get_coordinate_series(real)

        pixel = []
        for p in real_2d:
            pixel += [w[0] * p[..., order[0]] + b[0],
                      w[1] * p[..., order[1]] + b[1]]

        pixel = np.stack(pixel, axis=-1)

        if integer:
            pixel = pixel.astype(np.int32)
        return pixel

    def rescale(self, f: np.ndarray):
        if (s := self.info.datasetInfo.scale_vis) > 1:
            x, y = f.shape[:2]
            f = cv2.resize(f, (int(y/s), int(x/s)))
        return f

    def draw(self, agent: Agent,
             frames: list[int],
             save_name: str,
             draw_dis=False,
             interp=True,
             save_as_images=False):
        """
        Draw trajectories on the video.

        :param agent: The agent object (`Agent`) to visualize.
        :param frames: A list frames of current observation steps.
        :param save_name: The name to save the output video, which does not contain
            the file format.
        :param draw_dis: Choose if to draw trajectories as
            distributions or single dots.
        :param interp: Choose whether to draw the full video or only
            draw on the sampled time steps.
        :param save_as_images: Choose if to save as an image or a video clip.
        """
        f = None
        state = None

        # draw on video frames
        if self.video_capture:
            f = self.get_image(frames[0])

        if f is not None:
            state = DRAW_ON_VIDEO
            vis_func = self.vis
            text_func = self.text
            integer = True

            if not save_as_images:
                video_shape = (f.shape[1], f.shape[0])
                VideoWriter = cv2.VideoWriter(save_name + '.mp4',
                                              cv2.VideoWriter_fourcc(*'mp4v'),
                                              self.info.paras[1],
                                              video_shape)

        elif (f is None) and (self.scene_image is not None):
            state = DRAW_ON_IMAGE
            vis_func = self.vis
            text_func = self.text
            integer = True
            f = self.scene_image

        else:
            state = DRAW_ON_EMPTY
            integer = False
            vis_func = self._visualization_plt
            text_func = self._put_text_plt
            plt.figure()

        if f is not None:
            f_empty = np.zeros((f.shape[0], f.shape[1], 4))
        else:
            f_empty = None

        # interpolate frames
        if interp:
            frames = np.arange(frames[0], frames[-1]+1)

        agent_frames = agent.frames
        obs_len = agent.obs_length
        obs, pred, gt, nei = self.get_trajectories(agent, integer)

        if save_as_images:
            frame = frames[0]
            f = vis_func(f, obs, gt, pred, nei, draw_dis=draw_dis)
            f = text_func(f, self.get_text(frame, agent))

            if DRAW_TEXT_IN_IMAGES:
                cv2.imwrite(save_name + f'_{frame}.jpg', f)
            return

        f_pred = vis_func(f_empty, pred=pred, draw_dis=draw_dis)
        f_others = np.zeros_like(f_pred)

        for frame in SecondaryBar(frames,
                                  manager=self.manager,
                                  desc='Processing frames...'):
            # get a new scene image
            if state == DRAW_ON_VIDEO:
                f = self.get_image(frame)
            elif state == DRAW_ON_IMAGE:
                f = self.scene_image.copy()
            elif state == DRAW_ON_EMPTY:
                f = None
            else:
                raise ValueError(state)

            if frame in agent_frames:
                step = agent_frames.index(frame)
                if step < obs_len:
                    start, end = [max(0, step-1), step+1]
                    f_others = vis_func(f_others, obs=obs[start:end])
                else:
                    step -= obs_len
                    start, end = [max(0, step-1), step+1]
                    f_others = vis_func(f_others, gt=gt[start:end])

            # draw predictions
            if frame > agent_frames[obs_len-1]:
                f = vis_func(f, background=f_pred)

            # draw observations and groundtruths
            f = vis_func(f, background=f_others)

            if DRAW_TEXT_IN_VIDEOS:
                f = text_func(f, self.get_text(frame, agent))

            VideoWriter.write(f)

    def vis(self, source: np.ndarray,
            obs=None, gt=None, pred=None,
            neighbor=None,
            background: np.ndarray = None,
            draw_dis: int = 0):
        """
        Draw one agent's observations, predictions, and groundtruths.

        :param source: The image file.
        :param obs: (optional) The observations in *pixel* scale.
        :param gt: (optional) The ground truth in *pixel* scale.
        :param pred: (optional) The predictions in *pixel* scale,\
            shape = `(K, steps, dim)`.
        :param neighbor: (optional) The observed neighbors' positions\
             in *pixel* scale, shape = `(batch, dim)`.
        :param draw_distribution: Controls whether to draw as a distribution.
        :param alpha: The alpha channel coefficient.
        """
        f = np.zeros([source.shape[0], source.shape[1], 4])

        # draw predicted trajectories
        if pred is not None:
            if draw_dis:
                f = self.helper.draw_dis(f, pred, self.dis_file, alpha=0.8)
            else:
                for pred_k in pred:
                    f = self.helper.draw_traj(
                        f, pred_k, self.pred_file,
                        draw_lines=False,
                        color=255 * np.random.rand(3))

        if neighbor is not None:
            f = self.helper.draw_traj(
                f, neighbor, self.neighbor_file, draw_lines=False)

        if obs is not None:
            f = self.helper.draw_traj(f, obs, self.obs_file)

        if gt is not None:
            f = self.helper.draw_traj(f, gt, self.gt_file)

        # draw the background image
        if background is not None:
            f = ADD(background, f, [f.shape[1]//2, f.shape[0]//2])

        # add the original image
        f = ADD(source, f, [f.shape[1]//2, f.shape[0]//2])
        return f

    def text(self, f: np.ndarray, texts: list[str]) -> np.ndarray:
        """
        Put text on one image
        """
        for index, text in enumerate(texts):
            f = cv2.putText(f, text,
                            org=(10 + 3, 40 + index * 30 + 3),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=0.9,
                            color=(0, 0, 0),
                            thickness=2)

            f = cv2.putText(f, text,
                            org=(10, 40 + index * 30),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=0.9,
                            color=(255, 255, 255),
                            thickness=2)

        return f

    def _visualization_plt(self, f,
                           obs=None, gt=None, pred=None,
                           neighbor=None,
                           **kwargs):

        if obs is not None:
            for p in np.transpose(obs, [1, 0, 2]):
                plt.scatter(p.T[0], p.T[1], c='#287AFB')

        # if neighbor is not None:
        if None:
            for p in np.transpose(neighbor, [1, 0, 2]):
                plt.scatter(p.T[0], p.T[1], c='#CA15A9')

        if gt is not None:
            for p in np.transpose(gt, [1, 0, 2]):
                plt.scatter(p.T[0], p.T[1], c='#4CEDA7')

        if pred is not None:
            for p in np.transpose(pred, [1, 0, 2]):
                plt.scatter(p.T[0], p.T[1], c='#F5E25F')

        plt.axis('equal')

    def _put_text_plt(self, f: np.ndarray, texts: list[str]):
        plt.title(', '.join(texts))


def __draw_single_boundingbox(source, box: np.ndarray, png_file,
                              color, width, alpha):
    """
    The shape of `box` is `(4)`.
    """
    (y1, x1, y2, x2) = box[:4]
    cv2.rectangle(img=source,
                  pt1=(x1, y1),
                  pt2=(x2, y2),
                  color=color,
                  thickness=width)
    return source


def __draw_traj_boundingboxes(source, trajs, png_file,
                              color, width, alpha):
    """
    The shape of `trajs` is `(steps, 4)`.
    """
    for box in trajs:
        source = __draw_single_boundingbox(
            source, box, png_file, color, width, alpha)

    # draw center point
    source[:, :, 3] = alpha * 255 * source[:, :, 0]/color[0]
    for box in trajs:
        source = ADD(source, png_file,
                     ((box[1]+box[3])//2, (box[0]+box[2])//2))

    return source
