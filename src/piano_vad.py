# SPDX-License-Identifier: Apache-2.0
# Includes modifications to ByteDance's piano_transcription implementation.

import numpy as np


def note_detection_with_onset_offset_regress(frame_output, onset_output,
    onset_shift_output, offset_output, offset_shift_output, velocity_output,
    frame_threshold):
    """Process prediction matrices to note events information.
    First, detect onsets with onset outputs. Then, detect offsets
    with frame and offset outputs.

    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,)
      onset_shift_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      velocity_output: (frames_num,)
      frame_threshold: float

    Returns:
      output_tuples: list of [bgn, fin, onset_shift, offset_shift, normalized_velocity],
      e.g., [
        [1821, 1909, 0.47498, 0.3048533, 0.72119445],
        [1909, 1947, 0.30730522, -0.45764327, 0.64200014],
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    frames_num = onset_output.shape[0]
    for i in range(frames_num):
        if onset_output[i] == 1:
            """Onset detected"""
            if bgn is not None:
                """Consecutive onsets. E.g., pedal is not released, but two
                consecutive notes being played."""
                fin = max(i - 1, 0)
                output_tuples.append([bgn, fin, onset_shift_output[bgn],
                    0, velocity_output[bgn]])
                frame_disappear, offset_occur = None, None
            bgn = i

        if bgn is not None and i > bgn:
            """If onset found, then search offset"""
            if frame_output[i] <= frame_threshold and frame_disappear is None:
                """Frame disappear detected"""
                frame_disappear = i

            if offset_output[i] == 1 and offset_occur is None:
                """Offset detected"""
                offset_occur = i

            if frame_disappear is not None:
                if offset_occur is not None and offset_occur - bgn > frame_disappear - offset_occur:
                    """bgn --------- offset_occur --- frame_disappear"""
                    fin = offset_occur
                else:
                    """bgn --- offset_occur --------- frame_disappear"""
                    fin = frame_disappear
                output_tuples.append([bgn, fin, onset_shift_output[bgn],
                    offset_shift_output[fin], velocity_output[bgn]])
                bgn, frame_disappear, offset_occur = None, None, None

            if bgn is not None and i - bgn >= 600:
                """Maximum note duration reached."""
                fin = i
                output_tuples.append([bgn, fin, onset_shift_output[bgn],
                    offset_shift_output[fin], velocity_output[bgn]])
                bgn, frame_disappear, offset_occur = None, None, None

    # A held final chord ends at the final observed frame. An onset first seen
    # on that boundary has no supported duration inside the audio and is
    # discarded instead of emitting a zero-length or beyond-audio event.
    if bgn is not None and frames_num > 0 and bgn < frames_num - 1:
        fin = offset_occur if offset_occur is not None else frames_num - 1
        offset_shift = offset_shift_output[fin] if offset_occur is not None else 0.0
        output_tuples.append([bgn, fin, onset_shift_output[bgn],
            offset_shift, velocity_output[bgn]])

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples


def pedal_detection_with_onset_offset_regress(frame_output, offset_output,
    offset_shift_output, frame_threshold):
    """Process prediction array to pedal events information.

    Args:
      frame_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      frame_threshold: float

    Returns:
      output_tuples: list of [bgn, fin, onset_shift, offset_shift],
      e.g., [
        [1821, 1909, 0.4749851, 0.3048533],
        [1909, 1947, 0.30730522, -0.45764327],
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(1, frame_output.shape[0]):
        if frame_output[i] >= frame_threshold and frame_output[i] > frame_output[i - 1]:
            """Pedal onset detected"""
            if bgn:
                pass
            else:
                bgn = i

        if bgn and i > bgn:
            """If onset found, then search offset"""
            if frame_output[i] <= frame_threshold and not frame_disappear:
                """Frame disappear detected"""
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                """Offset detected"""
                offset_occur = i

            if offset_occur:
                fin = offset_occur
                output_tuples.append([bgn, fin, 0., offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None

            if frame_disappear and i - frame_disappear >= 10:
                """offset not detected but frame disappear"""
                fin = frame_disappear
                output_tuples.append([bgn, fin, 0., offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples


###### Google's onsets and frames post processing. Only used for comparison ######
def onsets_frames_note_detection(frame_output, onset_output, offset_output,
    velocity_output, threshold):
    """Process note prediction matrices into note events.

    ``onset_output`` starts a note. A subsequent nonzero ``offset_output`` or a
    frame probability at or below ``threshold`` ends it, whichever occurs
    first. A repeated onset closes the previous note at the repeated onset. A
    held note closes on the final observed frame; an onset first detected on
    that boundary is discarded because it has no supported positive duration.

    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,)
      offset_output: (frames_num,), binarized offset pulses. May be all zero.
      velocity_output: (frames_num,)
      threshold: float

    Returns:
      bgn_fin_pairs: list of [bgn, fin, velocity]. E.g.
        [[1821, 1909, 0.47498, 0.72119445],
         [1909, 1947, 0.30730522, 0.64200014],
         ...]
    """
    output_tuples = []

    loct = None
    frames_num = onset_output.shape[0]
    for i in range(frames_num):
        # Use onset_output is used to detect the presence of notes
        if onset_output[i] > threshold:
            if loct is not None:
                output_tuples.append([loct, i, velocity_output[loct]])
            loct = i
        if loct is not None and i > loct:
            frame_ended = frame_output[i] <= threshold
            explicit_offset = offset_output is not None and offset_output[i] > 0
            if explicit_offset or frame_ended:
                output_tuples.append([loct, i, velocity_output[loct]])
                loct = None

    if loct is not None and frames_num > 0 and loct < frames_num - 1:
        output_tuples.append([loct, frames_num - 1, velocity_output[loct]])

    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples


def onsets_frames_pedal_detection(frame_output, offset_output, frame_threshold):
    """Process pedal prediction matrices to pedal events information.

    Args:
      frame_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      frame_threshold: float

    Returns:
      output_tuples: list of [bgn, fin],
      e.g., [
        [1821, 1909],
        [1909, 1947],
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(1, frame_output.shape[0]):
        if frame_output[i] >= frame_threshold and frame_output[i] > frame_output[i - 1]:
            if bgn:
                pass
            else:
                bgn = i

        if bgn and i > bgn:
            """If onset found, then search offset"""
            if frame_output[i] <= frame_threshold and not frame_disappear:
                """Frame disappear detected"""
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                """Offset detected"""
                offset_occur = i

            if offset_occur:
                fin = offset_occur
                output_tuples.append([bgn, fin])
                bgn, frame_disappear, offset_occur = None, None, None

            if frame_disappear and i - frame_disappear >= 10:
                """offset not detected but frame disappear"""
                fin = frame_disappear
                output_tuples.append([bgn, fin])
                bgn, frame_disappear, offset_occur = None, None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples
