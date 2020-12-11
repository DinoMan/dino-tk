import ffmpeg
import scipy.io.wavfile as wav
import numpy as np
import torch
import tempfile
import os
from io import BytesIO
import warnings
import cv2

FACE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
              (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),  # chin
              (17, 18), (18, 19), (19, 20), (20, 21),  # right eyebrow
              (22, 23), (23, 24), (24, 25), (25, 26),  # left eyebrow
              (27, 28), (28, 29), (29, 30),  # nose bridge
              (31, 32), (32, 33), (33, 34), (34, 35),  # nose tip
              (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),  # right eye
              (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),  # left eye
              (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
              (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),  # outer mouth
              (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66),
              (66, 67), (67, 60)]  # inner mouth


def save_audio(path, audio, audio_rate=16000):
    if torch.is_tensor(audio):
        aud = audio.squeeze().detach().cpu().numpy()
    else:
        aud = audio.copy()  # Make a copy so that we don't alter the object

    aud = ((2 ** 15) * aud).astype(np.int16)
    wav.write(path, audio_rate, aud)


def save_video(path, video, fps=25, scale=2, audio=None, audio_rate=16000, ffmpeg_experimental=False):
    out_size = (scale * video.shape[-2], scale * video.shape[-1])
    video_path = "/tmp/" + next(tempfile._get_candidate_names()) + ".mp4"
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), out_size)
    if torch.is_tensor(video):
        vid = video.squeeze().detach().cpu().numpy()
    else:
        vid = video.copy()  # Make a copy so that we don't alter the object

    if np.min(vid) < 0:
        vid = 125 * vid + 125
    elif np.max(vid) <= 1:
        vid = 255 * vid

    for frame in vid:
        frame = np.rollaxis(frame, 0, 3)
        if scale != 1:
            frame = cv2.resize(frame, out_size)
        writer.write(frame.astype('uint8'))
    writer.release()

    inputs = [ffmpeg.input(path)['v']]

    if audio is not None:  # Save the audio file
        audio_path = "/tmp/" + next(tempfile._get_candidate_names()) + ".wav"
        save_audio(audio_path, audio, audio_rate)
        inputs += [ffmpeg.input(audio_path)['a']]

    try:
        if ffmpeg_experimental:
            out = ffmpeg.output(*inputs, path, strict='-2', loglevel="panic").overwrite_output()
        else:
            out = ffmpeg.output(*inputs, path, loglevel="panic").overwrite_output()
        out.run(quiet=True)
    except:
        return False

    if audio is not None and os.path.isfile(audio_path):
        os.remove(audio_path)
    if os.path.isfile(video_path):
        os.remove(video_path)

    return True


def video_to_stream(video, audio=None, fps=25, audio_rate=16000):
    temp_file = "/tmp/" + next(tempfile._get_candidate_names()) + ".mp4"
    save_video(temp_file, video, audio=audio, fps=fps, audio_rate=audio_rate)
    stream = BytesIO(open(temp_file, "rb").read())

    if os.path.isfile(temp_file):
        os.remove(temp_file)

    return stream


def save_joint_animation(path, points, edges, fps=25, audio=None, audio_rate=16000, colour=None, ffmpeg_experimental=False):
    if points.ndim == 3 and points.shape[2] > 3:
        warnings.warn("points have dimension larger than 3", RuntimeWarning)

    if edges == "face":
        edges = FACE_EDGES

    min_coord = np.min(points.reshape((-1, 2)), axis=0)
    max_coord = np.max(points.reshape((-1, 2)), axis=0)

    width = int(max_coord[0] - min_coord[0])
    height = int(max_coord[1] - min_coord[1])

    video_path = "/tmp/" + next(tempfile._get_candidate_names()) + ".mp4"
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (height, width))
    for frame in points:
        frame = frame - np.array([min_coord[0], min_coord[1]])
        canvas = np.ones((width, height, 3))
        canvas *= (255, 255, 255)  # canvas is by default white

        for node in frame:
            cv2.circle(canvas, (int(node[0]), int(node[1])), 2, colour, -1)

        for edge in edges:
            cv2.line(canvas, (int(frame[edge[0]][0]), int(frame[edge[0]][1])), (int(frame[edge[1]][0]), int(frame[edge[1]][1])), colour, 1)

        video.write(canvas.astype('uint8'))
    video.release()

    inputs = [ffmpeg.input(video_path)['v']]
    if audio is not None:  # Save the audio file
        audio_path = "/tmp/" + next(tempfile._get_candidate_names()) + ".wav"
        save_audio(audio_path, audio, audio_rate)
        inputs += [ffmpeg.input(audio_path)['a']]

    try:
        if ffmpeg_experimental:
            out = ffmpeg.output(*inputs, path, strict='-2', loglevel="panic").overwrite_output()
        else:
            out = ffmpeg.output(*inputs, path, loglevel="panic").overwrite_output()
        out.run(quiet=True)
    except:
        return False

    if audio is not None and os.path.isfile(audio_path):
        os.remove(audio_path)

    if os.path.isfile(video_path):
        os.remove(video_path)

    return True


def joint_animation_to_stream(points, edges, fps=25, audio=None, audio_rate=16000, colour=None):
    temp_file = "/tmp/" + next(tempfile._get_candidate_names()) + ".mp4"
    save_joint_animation(temp_file, points, edges, fps=fps, audio=audio, audio_rate=audio_rate, colour=colour)
    stream = BytesIO(open(temp_file, "rb").read())

    if os.path.isfile(temp_file):
        os.remove(temp_file)

    return stream