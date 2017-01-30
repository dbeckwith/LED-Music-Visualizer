import datetime


_timer = None
_start_time = None
_timer_stage = None

def timer(stage=None):
    global _timer, _start_time, _timer_stage
    t = datetime.datetime.now()
    if _start_time == None:
        _start_time = t
    if _timer != None:
        print('{} done in {}'.format(_timer_stage, t - _timer))
    if stage is None:
        print('Done in {}'.format(t - _start_time))
    else:
        _timer = t
        _timer_stage = stage
        print('{}...'.format(stage))
