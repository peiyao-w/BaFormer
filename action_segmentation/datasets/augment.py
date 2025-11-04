import random



def augment_crop(data, target):
    percent = random.uniform(0.8, 1)
    len_video = data.shape[2]
    len_video_clip = round(data.shape[2] * percent)
    # if len_video- len_video_clip == 0:
    #     start = 0
    # else:
    #     start = random.sample(range(0, len_video- len_video_clip +1), 1)[0]# +1
    # data = data[:,:,start: start + len_video_clip]
    # target = target[:, start: start + len_video_clip]
    # return data, target

    start = random.sample(range(0, len_video- len_video_clip +1), 1)[0]# +1
    data = data[:,:,start: start + len_video_clip]
    target = target[:, start: start + len_video_clip]
    return data, target

