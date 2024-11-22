from detect import run, parse_opt
from utils.general import check_requirements
from pathlib import Path

def birdEyeView(input_video):
    opt = parse_opt()
    opt.source = input_video
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))
    return Path("video.mp4")

input_video = "data/videos/demo-s.mp4"
output_video = birdEyeView(input_video)
