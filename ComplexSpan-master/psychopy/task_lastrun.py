#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on Thu Nov  2 15:15:30 2023
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'test_final'  # from the Builder filename that created this script
expInfo = {
    'participant': '',
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/wangerxiao/Desktop/ComplexSpan-master/ComplexSpan-master/psychopy/task_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=(1024, 768), fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Instrction_prac_memory" ---
    instructions_2 = visual.TextStim(win=win, name='instructions_2',
        text='Letters will now appear one by one.\nPlease remember them in the correct order of appearance.\nDo not say the letters out loud while recalling.\n\nPress space to continue.\n',
        font='Arial',
        pos=(0, 0), height=0.08, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_4 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "show_letters" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='',
        font='Arial',
        pos=(0, 0), height=0.2, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from code_6
    import random
    
    stimuli_letters = ['Z', 'Q', 'J', 'H', 'K', 'T', 'S', 'N' , 'R', 'X' ,'L', 'V']
    
    stimuli_count = 2 
    
    random.shuffle(stimuli_letters)
    
    chosen_stimuli = stimuli_letters[0:stimuli_count]
    correct_answer = ''.join(chosen_stimuli)
    loop_i = 0 #helps to specify which letters to print
    
    
    
    # --- Initialize components for Routine "recall" ---
    key_resp_5 = keyboard.Keyboard()
    text_feedback = visual.TextStim(win=win, name='text_feedback',
        text='',
        font='Arial',
        pos=(0, 0), height=0.15, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    # Run 'Begin Experiment' code from code_5
    total_memory_acc = 0
    memory_wrong = 0
    instructions_3 = visual.TextStim(win=win, name='instructions_3',
        text='',
        font='Arial',
        pos=(0, 0.25), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "instructions_prac_processing" ---
    instructions = visual.TextStim(win=win, name='instructions',
        text='You will now see a series of math equations, of which you will type in the correct answer.\n\nPlease also try to keep your score above 65%.\n\nPress space to continue.',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trials_processing" ---
    operation = visual.TextStim(win=win, name='operation',
        text='',
        font='Arial',
        pos=(0, 0.2), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard()
    operation_feedback = visual.TextStim(win=win, name='operation_feedback',
        text='',
        font='Arial',
        pos=(0, -0.25), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    # Run 'Begin Experiment' code from code_14
    RT_list = [] #empty list to hold values
    instructions_8 = visual.TextStim(win=win, name='instructions_8',
        text='',
        font='Arial',
        pos=(0, -0.2), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "feedback" ---
    # Run 'Begin Experiment' code from code
    msg = ''
    
    feedback_msg = visual.TextStim(win=win, name='feedback_msg',
        text='',
        font='Arial',
        pos=(0,0), height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "final_feedback" ---
    # Run 'Begin Experiment' code from code_2
    proc_msg = ''
    sumRT = 0
    sumdiff = 0
    feedback_msg_2 = visual.TextStim(win=win, name='feedback_msg_2',
        text='',
        font='Arial',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_3 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "instructions_complex_prac" ---
    instructions_4 = visual.TextStim(win=win, name='instructions_4',
        text='In this block, each math equation will be followed by a letter to be recalled.\n\nEquations will continually be presented unless you correctly answer them in the limited time given.\nPlease accurately respond to the equations and remember the letters presented after.\n\nYou will first start with some practice trials where feedback is provided.\n\nPress space to continue.',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_6 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trials_processing_prac" ---
    operation_3 = visual.TextStim(win=win, name='operation_3',
        text='',
        font='Arial',
        pos=(0, 0.2), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_11 = keyboard.Keyboard()
    operation_feedback_2 = visual.TextStim(win=win, name='operation_feedback_2',
        text='',
        font='Arial',
        pos=(0, -0.25), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    instructions_9 = visual.TextStim(win=win, name='instructions_9',
        text='',
        font='Arial',
        pos=(0, -0.2), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "feedback_quick" ---
    # Run 'Begin Experiment' code from code_15
    msg3 = ''
    loop_o = 0
    number_correct_2 = 0
    feedback_msg_4 = visual.TextStim(win=win, name='feedback_msg_4',
        text='',
        font='Arial',
        pos=(0, -0.2), height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    operation_4 = visual.TextStim(win=win, name='operation_4',
        text='',
        font='Arial',
        pos=(0, 0.2), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "show_letters_prac" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='',
        font='Arial',
        pos=(0, 0), height=0.2, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from code_7
    import random
    
    stimuli_letters = ['1', '2', '3', '4', '5', '6', '7', '8' , '9', '0']
    
    random.shuffle(stimuli_letters) #shuffle letters
    
    chosen_stimuli_3 = stimuli_letters[0:2] #choose 2 letters
    
    correct_answer_3 = ''.join(chosen_stimuli_3) #join 2 letters
    
    loop_i_3 = 0
    
    
    
    # --- Initialize components for Routine "recall_prac" ---
    key_resp_9 = keyboard.Keyboard()
    text_feedback_3 = visual.TextStim(win=win, name='text_feedback_3',
        text='',
        font='Arial',
        pos=(0, 0), height=0.15, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    instructions_6 = visual.TextStim(win=win, name='instructions_6',
        text='',
        font='Arial',
        pos=(0, 0.25), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "feedback_2" ---
    # Run 'Begin Experiment' code from code_12
    msg2 = ''
    prac_acc = 0
    feedback_msg_3 = visual.TextStim(win=win, name='feedback_msg_3',
        text='',
        font='Arial',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "instructions_complex" ---
    instructions_7 = visual.TextStim(win=win, name='instructions_7',
        text='Now you will start the actual task.\n\nAgain, each math equation will be followed by a letter to be recalled.\n\nEquations will continually be presented unless you correctly answer them in the limited time given.\nPlease accurately respond to the equations and remember the letters presented after.\nNo feedback will be provided on whether the letters are correct or not.\nTry not to say the letters out loud while recalling.\n\nPress space to continue.',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_10 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trials_processing_2" ---
    operation_2 = visual.TextStim(win=win, name='operation_2',
        text='',
        font='Arial',
        pos=(0, 0.2), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_7 = keyboard.Keyboard()
    operation_feedback_3 = visual.TextStim(win=win, name='operation_feedback_3',
        text='',
        font='Arial',
        pos=(0, -0.25), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    instructions_10 = visual.TextStim(win=win, name='instructions_10',
        text='',
        font='Arial',
        pos=(0, -0.2), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "feedback_quick_2" ---
    # Run 'Begin Experiment' code from code_16
    msg4 = ''
    loop_o_2 = 0
    number_correct_3 = 0
    feedback_msg_5 = visual.TextStim(win=win, name='feedback_msg_5',
        text='',
        font='Arial',
        pos=(0, -0.2), height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    operation_5 = visual.TextStim(win=win, name='operation_5',
        text='',
        font='Arial',
        pos=(0, 0.2), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "show_letters_2" ---
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Arial',
        pos=(0, 0), height=0.2, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from code_9
    import random
    
    stimuli_letters = ['Z', 'Q', 'J', 'H', 'K', 'T', 'S', 'N' , 'R', 'X' ,'L', 'V']
    
    span_count = 2
    
    random.shuffle(stimuli_letters) #shuffle letters
    
    chosen_stimuli_2 = stimuli_letters[0:span_count] #choose n letters
    
    correct_answer_2 = ''.join(chosen_stimuli_2) #join n letters
    
    loop_i_2 = 0
    
    
    
    # --- Initialize components for Routine "recall_2" ---
    key_resp_8 = keyboard.Keyboard()
    text_feedback_2 = visual.TextStim(win=win, name='text_feedback_2',
        text='',
        font='Arial',
        pos=(0, 0), height=0.15, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    # Run 'Begin Experiment' code from code_10
    total_complex_acc = 0
    complex_wrong = 0
    
    instructions_5 = visual.TextStim(win=win, name='instructions_5',
        text='',
        font='Arial',
        pos=(0, 0.25), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "finished" ---
    finish = visual.TextStim(win=win, name='finish',
        text='You have now come to the end of the task.\nPlease call the experimenter into the room.',
        font='Arial',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_12 = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "Instrction_prac_memory" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instrction_prac_memory.started', globalClock.getTime())
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # keep track of which components have finished
    Instrction_prac_memoryComponents = [instructions_2, key_resp_4]
    for thisComponent in Instrction_prac_memoryComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instrction_prac_memory" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_2* updates
        
        # if instructions_2 is starting this frame...
        if instructions_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            instructions_2.frameNStart = frameN  # exact frame index
            instructions_2.tStart = t  # local t and not account for scr refresh
            instructions_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_2.started')
            # update status
            instructions_2.status = STARTED
            instructions_2.setAutoDraw(True)
        
        # if instructions_2 is active this frame...
        if instructions_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_4* updates
        waitOnFlip = False
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_4.started')
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_4.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instrction_prac_memoryComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instrction_prac_memory" ---
    for thisComponent in Instrction_prac_memoryComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Instrction_prac_memory.stopped', globalClock.getTime())
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    thisExp.nextEntry()
    # the Routine "Instrction_prac_memory" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    next_level = data.TrialHandler(nReps=999, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='next_level')
    thisExp.addLoop(next_level)  # add the loop to the experiment
    thisNext_level = next_level.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisNext_level.rgb)
    if thisNext_level != None:
        for paramName in thisNext_level:
            globals()[paramName] = thisNext_level[paramName]
    
    for thisNext_level in next_level:
        currentLoop = next_level
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisNext_level.rgb)
        if thisNext_level != None:
            for paramName in thisNext_level:
                globals()[paramName] = thisNext_level[paramName]
        
        # set up handler to look after randomisation of conditions etc
        next_letter = data.TrialHandler(nReps=stimuli_count, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='next_letter')
        thisExp.addLoop(next_letter)  # add the loop to the experiment
        thisNext_letter = next_letter.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisNext_letter.rgb)
        if thisNext_letter != None:
            for paramName in thisNext_letter:
                globals()[paramName] = thisNext_letter[paramName]
        
        for thisNext_letter in next_letter:
            currentLoop = next_letter
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisNext_letter.rgb)
            if thisNext_letter != None:
                for paramName in thisNext_letter:
                    globals()[paramName] = thisNext_letter[paramName]
            
            # --- Prepare to start Routine "show_letters" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('show_letters.started', globalClock.getTime())
            text_2.setText(chosen_stimuli[loop_i])
            # Run 'Begin Routine' code from code_6
            loop_i += 1
            
            # keep track of which components have finished
            show_lettersComponents = [text_2]
            for thisComponent in show_lettersComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "show_letters" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_2* updates
                
                # if text_2 is starting this frame...
                if text_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    text_2.frameNStart = frameN  # exact frame index
                    text_2.tStart = t  # local t and not account for scr refresh
                    text_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_2.started')
                    # update status
                    text_2.status = STARTED
                    text_2.setAutoDraw(True)
                
                # if text_2 is active this frame...
                if text_2.status == STARTED:
                    # update params
                    pass
                
                # if text_2 is stopping this frame...
                if text_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_2.tStartRefresh + 0.8-frameTolerance:
                        # keep track of stop time/frame for later
                        text_2.tStop = t  # not accounting for scr refresh
                        text_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_2.stopped')
                        # update status
                        text_2.status = FINISHED
                        text_2.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in show_lettersComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "show_letters" ---
            for thisComponent in show_lettersComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('show_letters.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.300000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed stimuli_count repeats of 'next_letter'
        
        
        # --- Prepare to start Routine "recall" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('recall.started', globalClock.getTime())
        key_resp_5.keys = []
        key_resp_5.rt = []
        _key_resp_5_allKeys = []
        instructions_3.setText('Answer and press enter.')
        # keep track of which components have finished
        recallComponents = [key_resp_5, text_feedback, instructions_3]
        for thisComponent in recallComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recall" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *key_resp_5* updates
            waitOnFlip = False
            
            # if key_resp_5 is starting this frame...
            if key_resp_5.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                key_resp_5.frameNStart = frameN  # exact frame index
                key_resp_5.tStart = t  # local t and not account for scr refresh
                key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_5.started')
                # update status
                key_resp_5.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_5.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_5.getKeys(keyList=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z', 'backspace','return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_5_allKeys.extend(theseKeys)
                if len(_key_resp_5_allKeys):
                    key_resp_5.keys = [key.name for key in _key_resp_5_allKeys]  # storing all keys
                    key_resp_5.rt = [key.rt for key in _key_resp_5_allKeys]
                    key_resp_5.duration = [key.duration for key in _key_resp_5_allKeys]
            
            # *text_feedback* updates
            
            # if text_feedback is starting this frame...
            if text_feedback.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_feedback.frameNStart = frameN  # exact frame index
                text_feedback.tStart = t  # local t and not account for scr refresh
                text_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_feedback.started')
                # update status
                text_feedback.status = STARTED
                text_feedback.setAutoDraw(True)
            
            # if text_feedback is active this frame...
            if text_feedback.status == STARTED:
                # update params
                text_feedback.setText(screen_text
                , log=False)
            # Run 'Each Frame' code from code_5
            if("backspace" in key_resp_5.keys):
                key_resp_5.keys.remove("backspace") 
                
                if(len(key_resp_5.keys) > 0):
                    key_resp_5.keys.pop() #remove backspace
                    
            elif("return" in key_resp_5.keys):
                key_resp_5.keys.remove("return") #remove return
                
                if(len(key_resp_5.keys) > 0):
                    screen_text = ''.join(key_resp_5.keys)
                    thisExp.addData("simple_recall", screen_text) #store response in data file after entering
            
                continueRoutine = False
            
            if(len(key_resp_5.keys) > 20):
                key_resp_5.keys.pop() #prevent typing after 20 keys entered
            
            
            screen_text = ''.join(key_resp_5.keys)
            screen_text = screen_text.upper()
            
            # *instructions_3* updates
            
            # if instructions_3 is starting this frame...
            if instructions_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                instructions_3.frameNStart = frameN  # exact frame index
                instructions_3.tStart = t  # local t and not account for scr refresh
                instructions_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instructions_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instructions_3.started')
                # update status
                instructions_3.status = STARTED
                instructions_3.setAutoDraw(True)
            
            # if instructions_3 is active this frame...
            if instructions_3.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recallComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recall" ---
        for thisComponent in recallComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('recall.stopped', globalClock.getTime())
        # check responses
        if key_resp_5.keys in ['', [], None]:  # No response was made
            key_resp_5.keys = None
        next_level.addData('key_resp_5.keys',key_resp_5.keys)
        if key_resp_5.keys != None:  # we had a response
            next_level.addData('key_resp_5.rt', key_resp_5.rt)
            next_level.addData('key_resp_5.duration', key_resp_5.duration)
        # Run 'End Routine' code from code_5
        ##if there are 3 or more wrong responses in 5 or less trials, break
        
        if (''.join(key_resp_5.keys).upper() == correct_answer.upper()):
            total_memory_acc = total_memory_acc + 1
            thisExp.addData("simple_recall_corr", "1")
            thisExp.addData("simple_recall_corrkeys", correct_answer.upper())
        else:
            memory_wrong = memory_wrong + 1
            thisExp.addData("simple_recall_corr", "0")
            thisExp.addData("simple_recall_corrkeys", correct_answer.upper())
        
        
        if (memory_wrong >= 3):
            next_level.finished = True
            continueRoutine = False
        elif (total_memory_acc >= 2):
            stimuli_count += 1
            total_memory_acc = 0
            memory_wrong = 0
        
        random.shuffle(stimuli_letters)
        
        chosen_stimuli = stimuli_letters[0:stimuli_count]
        correct_answer = ''.join(chosen_stimuli)
        loop_i = 0 #reset letters to print
        
        
        # the Routine "recall" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 999 repeats of 'next_level'
    
    
    # --- Prepare to start Routine "instructions_prac_processing" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructions_prac_processing.started', globalClock.getTime())
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # keep track of which components have finished
    instructions_prac_processingComponents = [instructions, key_resp_2]
    for thisComponent in instructions_prac_processingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_prac_processing" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions* updates
        
        # if instructions is starting this frame...
        if instructions.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            instructions.frameNStart = frameN  # exact frame index
            instructions.tStart = t  # local t and not account for scr refresh
            instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions.started')
            # update status
            instructions.status = STARTED
            instructions.setAutoDraw(True)
        
        # if instructions is active this frame...
        if instructions.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_prac_processingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_prac_processing" ---
    for thisComponent in instructions_prac_processingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructions_prac_processing.stopped', globalClock.getTime())
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "instructions_prac_processing" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    repeat_practice = data.TrialHandler(nReps=999, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='repeat_practice')
    thisExp.addLoop(repeat_practice)  # add the loop to the experiment
    thisRepeat_practice = repeat_practice.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRepeat_practice.rgb)
    if thisRepeat_practice != None:
        for paramName in thisRepeat_practice:
            globals()[paramName] = thisRepeat_practice[paramName]
    
    for thisRepeat_practice in repeat_practice:
        currentLoop = repeat_practice
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisRepeat_practice.rgb)
        if thisRepeat_practice != None:
            for paramName in thisRepeat_practice:
                globals()[paramName] = thisRepeat_practice[paramName]
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler(nReps=1, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('../materials/processing_only_2.xlsx'),
            seed=None, name='trials')
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "trials_processing" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trials_processing.started', globalClock.getTime())
            operation.setText(Problem)
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            instructions_8.setText('Answer?')
            # keep track of which components have finished
            trials_processingComponents = [operation, key_resp, operation_feedback, instructions_8]
            for thisComponent in trials_processingComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trials_processing" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *operation* updates
                
                # if operation is starting this frame...
                if operation.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    operation.frameNStart = frameN  # exact frame index
                    operation.tStart = t  # local t and not account for scr refresh
                    operation.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(operation, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'operation.started')
                    # update status
                    operation.status = STARTED
                    operation.setAutoDraw(True)
                
                # if operation is active this frame...
                if operation.status == STARTED:
                    # update params
                    pass
                
                # *key_resp* updates
                waitOnFlip = False
                
                # if key_resp is starting this frame...
                if key_resp.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.started')
                    # update status
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = [key.name for key in _key_resp_allKeys]  # storing all keys
                        key_resp.rt = [key.rt for key in _key_resp_allKeys]
                        key_resp.duration = [key.duration for key in _key_resp_allKeys]
                        # was this correct?
                        if (key_resp.keys == str(Operation_answer)) or (key_resp.keys == Operation_answer):
                            key_resp.corr = 1
                        else:
                            key_resp.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # *operation_feedback* updates
                
                # if operation_feedback is starting this frame...
                if operation_feedback.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    operation_feedback.frameNStart = frameN  # exact frame index
                    operation_feedback.tStart = t  # local t and not account for scr refresh
                    operation_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(operation_feedback, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'operation_feedback.started')
                    # update status
                    operation_feedback.status = STARTED
                    operation_feedback.setAutoDraw(True)
                
                # if operation_feedback is active this frame...
                if operation_feedback.status == STARTED:
                    # update params
                    operation_feedback.setText(operation_text, log=False)
                # Run 'Each Frame' code from code_14
                if(len(key_resp.keys) > 0):
                    operation_text = ''.join(key_resp.keys)
                    thisExp.addData("digit_answer", operation_text) #store response in data file after entering
                
                    continueRoutine = False
                
                operation_text = ''.join(key_resp.keys)
                
                
                # *instructions_8* updates
                
                # if instructions_8 is starting this frame...
                if instructions_8.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    instructions_8.frameNStart = frameN  # exact frame index
                    instructions_8.tStart = t  # local t and not account for scr refresh
                    instructions_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(instructions_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'instructions_8.started')
                    # update status
                    instructions_8.status = STARTED
                    instructions_8.setAutoDraw(True)
                
                # if instructions_8 is active this frame...
                if instructions_8.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trials_processingComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trials_processing" ---
            for thisComponent in trials_processingComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trials_processing.stopped', globalClock.getTime())
            # check responses
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
                # was no response the correct answer?!
                if str(Operation_answer).lower() == 'none':
                   key_resp.corr = 1;  # correct non-response
                else:
                   key_resp.corr = 0;  # failed to respond (incorrectly)
            # store data for trials (TrialHandler)
            trials.addData('key_resp.keys',key_resp.keys)
            trials.addData('key_resp.corr', key_resp.corr)
            if key_resp.keys != None:  # we had a response
                trials.addData('key_resp.rt', key_resp.rt)
                trials.addData('key_resp.duration', key_resp.duration)
            # Run 'End Routine' code from code_14
            key_resp.rt = float(''.join([str(x) for x in key_resp.rt]))
            
            RT_list.append(key_resp.rt)
            thisExp.addData('RT_list', RT_list)
            # the Routine "trials_processing" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "feedback" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('feedback.started', globalClock.getTime())
            # Run 'Begin Routine' code from code
            if trials.thisN == 0:
                number_correct = 0
            
            if (''.join(key_resp.keys) == str(Operation_answer)):
                number_correct = number_correct + 1
                msg='Correct! \nScore = %.1f%%' %(number_correct*100/trials.nTotal)
                msgColor = 'green'
            else:
                msg='Oops! That was wrong \nScore = %.1f%%' %(number_correct*100/trials.nTotal)
                msgColor = 'red'
            feedback_msg.setColor(msgColor, colorSpace='rgb')
            feedback_msg.setText(msg)
            # keep track of which components have finished
            feedbackComponents = [feedback_msg]
            for thisComponent in feedbackComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "feedback" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from code
                
                
                
                
                
                # *feedback_msg* updates
                
                # if feedback_msg is starting this frame...
                if feedback_msg.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    feedback_msg.frameNStart = frameN  # exact frame index
                    feedback_msg.tStart = t  # local t and not account for scr refresh
                    feedback_msg.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(feedback_msg, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_msg.started')
                    # update status
                    feedback_msg.status = STARTED
                    feedback_msg.setAutoDraw(True)
                
                # if feedback_msg is active this frame...
                if feedback_msg.status == STARTED:
                    # update params
                    pass
                
                # if feedback_msg is stopping this frame...
                if feedback_msg.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > feedback_msg.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        feedback_msg.tStop = t  # not accounting for scr refresh
                        feedback_msg.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'feedback_msg.stopped')
                        # update status
                        feedback_msg.status = FINISHED
                        feedback_msg.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in feedbackComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "feedback" ---
            for thisComponent in feedbackComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('feedback.stopped', globalClock.getTime())
            # Run 'End Routine' code from code
            
            
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.500000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1 repeats of 'trials'
        
        
        # --- Prepare to start Routine "final_feedback" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('final_feedback.started', globalClock.getTime())
        # Run 'Begin Routine' code from code_2
        if trials.thisN == trials.nTotal:
            if (number_correct/(trials.nTotal)) >= 0.65:
                proc_msg = 'Congratulations, you answered correctly on above 65% of the trials. Press space to continue.'
                repeat_practice.finished = True
                thisExp.addData("processing_score", (number_correct/(trials.nTotal)))
            else:
                proc_msg = 'You failed to answer correctly on at least 65% of the trials, please try again until you reach the passing rate of 65%. Press space to start.'
                repeat_practice.finished = False 
        
        
        for x in RT_list:
            sumRT = sumRT + x
        
        thisExp.addData('sumRT', sumRT)
        meanRT = sumRT/15
        thisExp.addData('meanRT', meanRT)
        
        for x in RT_list:
            sumdiff = sumdiff + ((x - meanRT)**2)
        
        thisExp.addData('sumdiff', sumdiff)
        sd = (sumdiff/15) ** 0.5
        thisExp.addData('sd', sd)
        
        RT = meanRT + (2.58*sd)
        thisExp.addData('RT', RT)
        
        feedback_msg_2.setText(proc_msg)
        key_resp_3.keys = []
        key_resp_3.rt = []
        _key_resp_3_allKeys = []
        # keep track of which components have finished
        final_feedbackComponents = [feedback_msg_2, key_resp_3]
        for thisComponent in final_feedbackComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "final_feedback" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_msg_2* updates
            
            # if feedback_msg_2 is starting this frame...
            if feedback_msg_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                feedback_msg_2.frameNStart = frameN  # exact frame index
                feedback_msg_2.tStart = t  # local t and not account for scr refresh
                feedback_msg_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_msg_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_msg_2.started')
                # update status
                feedback_msg_2.status = STARTED
                feedback_msg_2.setAutoDraw(True)
            
            # if feedback_msg_2 is active this frame...
            if feedback_msg_2.status == STARTED:
                # update params
                pass
            
            # *key_resp_3* updates
            waitOnFlip = False
            
            # if key_resp_3 is starting this frame...
            if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                key_resp_3.frameNStart = frameN  # exact frame index
                key_resp_3.tStart = t  # local t and not account for scr refresh
                key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_3.started')
                # update status
                key_resp_3.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            if key_resp_3.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_3_allKeys.extend(theseKeys)
                if len(_key_resp_3_allKeys):
                    key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                    key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                    key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in final_feedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "final_feedback" ---
        for thisComponent in final_feedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('final_feedback.stopped', globalClock.getTime())
        # check responses
        if key_resp_3.keys in ['', [], None]:  # No response was made
            key_resp_3.keys = None
        repeat_practice.addData('key_resp_3.keys',key_resp_3.keys)
        if key_resp_3.keys != None:  # we had a response
            repeat_practice.addData('key_resp_3.rt', key_resp_3.rt)
            repeat_practice.addData('key_resp_3.duration', key_resp_3.duration)
        # the Routine "final_feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 999 repeats of 'repeat_practice'
    
    
    # --- Prepare to start Routine "instructions_complex_prac" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructions_complex_prac.started', globalClock.getTime())
    key_resp_6.keys = []
    key_resp_6.rt = []
    _key_resp_6_allKeys = []
    # keep track of which components have finished
    instructions_complex_pracComponents = [instructions_4, key_resp_6]
    for thisComponent in instructions_complex_pracComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_complex_prac" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_4* updates
        
        # if instructions_4 is starting this frame...
        if instructions_4.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            instructions_4.frameNStart = frameN  # exact frame index
            instructions_4.tStart = t  # local t and not account for scr refresh
            instructions_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_4.started')
            # update status
            instructions_4.status = STARTED
            instructions_4.setAutoDraw(True)
        
        # if instructions_4 is active this frame...
        if instructions_4.status == STARTED:
            # update params
            pass
        
        # *key_resp_6* updates
        waitOnFlip = False
        
        # if key_resp_6 is starting this frame...
        if key_resp_6.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            key_resp_6.frameNStart = frameN  # exact frame index
            key_resp_6.tStart = t  # local t and not account for scr refresh
            key_resp_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_6.started')
            # update status
            key_resp_6.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_6.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_6.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_6.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_6_allKeys.extend(theseKeys)
            if len(_key_resp_6_allKeys):
                key_resp_6.keys = _key_resp_6_allKeys[-1].name  # just the last key pressed
                key_resp_6.rt = _key_resp_6_allKeys[-1].rt
                key_resp_6.duration = _key_resp_6_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_complex_pracComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_complex_prac" ---
    for thisComponent in instructions_complex_pracComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructions_complex_prac.stopped', globalClock.getTime())
    # check responses
    if key_resp_6.keys in ['', [], None]:  # No response was made
        key_resp_6.keys = None
    thisExp.addData('key_resp_6.keys',key_resp_6.keys)
    if key_resp_6.keys != None:  # we had a response
        thisExp.addData('key_resp_6.rt', key_resp_6.rt)
        thisExp.addData('key_resp_6.duration', key_resp_6.duration)
    thisExp.nextEntry()
    # the Routine "instructions_complex_prac" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    span_recall_prac = data.TrialHandler(nReps=999, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='span_recall_prac')
    thisExp.addLoop(span_recall_prac)  # add the loop to the experiment
    thisSpan_recall_prac = span_recall_prac.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisSpan_recall_prac.rgb)
    if thisSpan_recall_prac != None:
        for paramName in thisSpan_recall_prac:
            globals()[paramName] = thisSpan_recall_prac[paramName]
    
    for thisSpan_recall_prac in span_recall_prac:
        currentLoop = span_recall_prac
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisSpan_recall_prac.rgb)
        if thisSpan_recall_prac != None:
            for paramName in thisSpan_recall_prac:
                globals()[paramName] = thisSpan_recall_prac[paramName]
        
        # set up handler to look after randomisation of conditions etc
        span_size_prac = data.TrialHandler(nReps=2, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='span_size_prac')
        thisExp.addLoop(span_size_prac)  # add the loop to the experiment
        thisSpan_size_prac = span_size_prac.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisSpan_size_prac.rgb)
        if thisSpan_size_prac != None:
            for paramName in thisSpan_size_prac:
                globals()[paramName] = thisSpan_size_prac[paramName]
        
        for thisSpan_size_prac in span_size_prac:
            currentLoop = span_size_prac
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisSpan_size_prac.rgb)
            if thisSpan_size_prac != None:
                for paramName in thisSpan_size_prac:
                    globals()[paramName] = thisSpan_size_prac[paramName]
            
            # set up handler to look after randomisation of conditions etc
            repeat_if_wrong_prac = data.TrialHandler(nReps=999, method='random', 
                extraInfo=expInfo, originPath=-1,
                trialList=data.importConditions('../materials/processing_only_2.xlsx'),
                seed=None, name='repeat_if_wrong_prac')
            thisExp.addLoop(repeat_if_wrong_prac)  # add the loop to the experiment
            thisRepeat_if_wrong_prac = repeat_if_wrong_prac.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisRepeat_if_wrong_prac.rgb)
            if thisRepeat_if_wrong_prac != None:
                for paramName in thisRepeat_if_wrong_prac:
                    globals()[paramName] = thisRepeat_if_wrong_prac[paramName]
            
            for thisRepeat_if_wrong_prac in repeat_if_wrong_prac:
                currentLoop = repeat_if_wrong_prac
                thisExp.timestampOnFlip(win, 'thisRow.t')
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        inputs=inputs, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisRepeat_if_wrong_prac.rgb)
                if thisRepeat_if_wrong_prac != None:
                    for paramName in thisRepeat_if_wrong_prac:
                        globals()[paramName] = thisRepeat_if_wrong_prac[paramName]
                
                # --- Prepare to start Routine "trials_processing_prac" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('trials_processing_prac.started', globalClock.getTime())
                operation_3.setText(Problem
                )
                key_resp_11.keys = []
                key_resp_11.rt = []
                _key_resp_11_allKeys = []
                instructions_9.setText('Answer?')
                # keep track of which components have finished
                trials_processing_pracComponents = [operation_3, key_resp_11, operation_feedback_2, instructions_9]
                for thisComponent in trials_processing_pracComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "trials_processing_prac" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *operation_3* updates
                    
                    # if operation_3 is starting this frame...
                    if operation_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                        # keep track of start time/frame for later
                        operation_3.frameNStart = frameN  # exact frame index
                        operation_3.tStart = t  # local t and not account for scr refresh
                        operation_3.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(operation_3, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'operation_3.started')
                        # update status
                        operation_3.status = STARTED
                        operation_3.setAutoDraw(True)
                    
                    # if operation_3 is active this frame...
                    if operation_3.status == STARTED:
                        # update params
                        pass
                    
                    # if operation_3 is stopping this frame...
                    if operation_3.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > operation_3.tStartRefresh + RT-frameTolerance:
                            # keep track of stop time/frame for later
                            operation_3.tStop = t  # not accounting for scr refresh
                            operation_3.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'operation_3.stopped')
                            # update status
                            operation_3.status = FINISHED
                            operation_3.setAutoDraw(False)
                    
                    # *key_resp_11* updates
                    waitOnFlip = False
                    
                    # if key_resp_11 is starting this frame...
                    if key_resp_11.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_11.frameNStart = frameN  # exact frame index
                        key_resp_11.tStart = t  # local t and not account for scr refresh
                        key_resp_11.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_11, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_11.started')
                        # update status
                        key_resp_11.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(key_resp_11.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(key_resp_11.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if key_resp_11 is stopping this frame...
                    if key_resp_11.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_resp_11.tStartRefresh + RT-frameTolerance:
                            # keep track of stop time/frame for later
                            key_resp_11.tStop = t  # not accounting for scr refresh
                            key_resp_11.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'key_resp_11.stopped')
                            # update status
                            key_resp_11.status = FINISHED
                            key_resp_11.status = FINISHED
                    if key_resp_11.status == STARTED and not waitOnFlip:
                        theseKeys = key_resp_11.getKeys(keyList=['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_11_allKeys.extend(theseKeys)
                        if len(_key_resp_11_allKeys):
                            key_resp_11.keys = [key.name for key in _key_resp_11_allKeys]  # storing all keys
                            key_resp_11.rt = [key.rt for key in _key_resp_11_allKeys]
                            key_resp_11.duration = [key.duration for key in _key_resp_11_allKeys]
                            # was this correct?
                            if (key_resp_11.keys == str(Operation_answer)) or (key_resp_11.keys == Operation_answer):
                                key_resp_11.corr = 1
                            else:
                                key_resp_11.corr = 0
                            # a response ends the routine
                            continueRoutine = False
                    
                    # *operation_feedback_2* updates
                    
                    # if operation_feedback_2 is starting this frame...
                    if operation_feedback_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                        # keep track of start time/frame for later
                        operation_feedback_2.frameNStart = frameN  # exact frame index
                        operation_feedback_2.tStart = t  # local t and not account for scr refresh
                        operation_feedback_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(operation_feedback_2, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'operation_feedback_2.started')
                        # update status
                        operation_feedback_2.status = STARTED
                        operation_feedback_2.setAutoDraw(True)
                    
                    # if operation_feedback_2 is active this frame...
                    if operation_feedback_2.status == STARTED:
                        # update params
                        operation_feedback_2.setText(operation_text_2, log=False)
                    
                    # if operation_feedback_2 is stopping this frame...
                    if operation_feedback_2.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > operation_feedback_2.tStartRefresh + RT-frameTolerance:
                            # keep track of stop time/frame for later
                            operation_feedback_2.tStop = t  # not accounting for scr refresh
                            operation_feedback_2.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'operation_feedback_2.stopped')
                            # update status
                            operation_feedback_2.status = FINISHED
                            operation_feedback_2.setAutoDraw(False)
                    # Run 'Each Frame' code from code_13
                    if(len(key_resp_11.keys) > 0):
                        operation_text_2 = ''.join(key_resp_11.keys)
                        thisExp.addData("digit_answer", operation_text_2) #store response in data file after entering
                    
                        continueRoutine = False
                    
                    operation_text_2 = ''.join(key_resp_11.keys)
                    
                    # *instructions_9* updates
                    
                    # if instructions_9 is starting this frame...
                    if instructions_9.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                        # keep track of start time/frame for later
                        instructions_9.frameNStart = frameN  # exact frame index
                        instructions_9.tStart = t  # local t and not account for scr refresh
                        instructions_9.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(instructions_9, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'instructions_9.started')
                        # update status
                        instructions_9.status = STARTED
                        instructions_9.setAutoDraw(True)
                    
                    # if instructions_9 is active this frame...
                    if instructions_9.status == STARTED:
                        # update params
                        pass
                    
                    # if instructions_9 is stopping this frame...
                    if instructions_9.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > instructions_9.tStartRefresh + RT-frameTolerance:
                            # keep track of stop time/frame for later
                            instructions_9.tStop = t  # not accounting for scr refresh
                            instructions_9.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'instructions_9.stopped')
                            # update status
                            instructions_9.status = FINISHED
                            instructions_9.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, inputs=inputs, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in trials_processing_pracComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "trials_processing_prac" ---
                for thisComponent in trials_processing_pracComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('trials_processing_prac.stopped', globalClock.getTime())
                # check responses
                if key_resp_11.keys in ['', [], None]:  # No response was made
                    key_resp_11.keys = None
                    # was no response the correct answer?!
                    if str(Operation_answer).lower() == 'none':
                       key_resp_11.corr = 1;  # correct non-response
                    else:
                       key_resp_11.corr = 0;  # failed to respond (incorrectly)
                # store data for repeat_if_wrong_prac (TrialHandler)
                repeat_if_wrong_prac.addData('key_resp_11.keys',key_resp_11.keys)
                repeat_if_wrong_prac.addData('key_resp_11.corr', key_resp_11.corr)
                if key_resp_11.keys != None:  # we had a response
                    repeat_if_wrong_prac.addData('key_resp_11.rt', key_resp_11.rt)
                    repeat_if_wrong_prac.addData('key_resp_11.duration', key_resp_11.duration)
                # the Routine "trials_processing_prac" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "feedback_quick" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('feedback_quick.started', globalClock.getTime())
                # Run 'Begin Routine' code from code_15
                loop_o += 1
                
                if not key_resp_11.keys: #if no entry
                    continueRoutine = True
                    msg3 ='Be faster! \nScore = %.1f%%' %(number_correct_2*100/loop_o)
                    msg3Color = 'red'
                    thisExp.addData("operation_correct", 0)
                    repeat_if_wrong_prac.finished = False
                else:
                    if (''.join(key_resp_11.keys) == str(Operation_answer)):
                        number_correct_2 = number_correct_2 + 1
                        msg3 ='Correct! \nScore = %.1f%%' %(number_correct_2*100/loop_o)
                        msg3Color = 'green'
                        thisExp.addData("operation_correct", 1)
                        repeat_if_wrong_prac.finished = True 
                    else:
                        msg3 ='Oops! That was wrong \nScore = %.1f%%' %(number_correct_2*100/loop_o)
                        msg3Color = 'red'
                        thisExp.addData("operation_correct", 0)
                        repeat_if_wrong_prac.finished = False
                
                
                
                
                feedback_msg_4.setColor(msg3Color, colorSpace='rgb')
                feedback_msg_4.setText(msg3)
                operation_4.setText(Problem)
                # keep track of which components have finished
                feedback_quickComponents = [feedback_msg_4, operation_4]
                for thisComponent in feedback_quickComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "feedback_quick" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 0.8:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *feedback_msg_4* updates
                    
                    # if feedback_msg_4 is starting this frame...
                    if feedback_msg_4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        feedback_msg_4.frameNStart = frameN  # exact frame index
                        feedback_msg_4.tStart = t  # local t and not account for scr refresh
                        feedback_msg_4.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(feedback_msg_4, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'feedback_msg_4.started')
                        # update status
                        feedback_msg_4.status = STARTED
                        feedback_msg_4.setAutoDraw(True)
                    
                    # if feedback_msg_4 is active this frame...
                    if feedback_msg_4.status == STARTED:
                        # update params
                        pass
                    
                    # if feedback_msg_4 is stopping this frame...
                    if feedback_msg_4.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > feedback_msg_4.tStartRefresh + 0.8-frameTolerance:
                            # keep track of stop time/frame for later
                            feedback_msg_4.tStop = t  # not accounting for scr refresh
                            feedback_msg_4.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'feedback_msg_4.stopped')
                            # update status
                            feedback_msg_4.status = FINISHED
                            feedback_msg_4.setAutoDraw(False)
                    
                    # *operation_4* updates
                    
                    # if operation_4 is starting this frame...
                    if operation_4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        operation_4.frameNStart = frameN  # exact frame index
                        operation_4.tStart = t  # local t and not account for scr refresh
                        operation_4.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(operation_4, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'operation_4.started')
                        # update status
                        operation_4.status = STARTED
                        operation_4.setAutoDraw(True)
                    
                    # if operation_4 is active this frame...
                    if operation_4.status == STARTED:
                        # update params
                        pass
                    
                    # if operation_4 is stopping this frame...
                    if operation_4.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > operation_4.tStartRefresh + 0.8-frameTolerance:
                            # keep track of stop time/frame for later
                            operation_4.tStop = t  # not accounting for scr refresh
                            operation_4.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'operation_4.stopped')
                            # update status
                            operation_4.status = FINISHED
                            operation_4.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, inputs=inputs, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in feedback_quickComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "feedback_quick" ---
                for thisComponent in feedback_quickComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('feedback_quick.stopped', globalClock.getTime())
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-0.800000)
                thisExp.nextEntry()
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed 999 repeats of 'repeat_if_wrong_prac'
            
            
            # --- Prepare to start Routine "show_letters_prac" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('show_letters_prac.started', globalClock.getTime())
            text_3.setText(chosen_stimuli_3[loop_i_3])
            # Run 'Begin Routine' code from code_7
            loop_i_3 += 1
            
            # keep track of which components have finished
            show_letters_pracComponents = [text_3]
            for thisComponent in show_letters_pracComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "show_letters_prac" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_3* updates
                
                # if text_3 is starting this frame...
                if text_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    text_3.frameNStart = frameN  # exact frame index
                    text_3.tStart = t  # local t and not account for scr refresh
                    text_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_3.started')
                    # update status
                    text_3.status = STARTED
                    text_3.setAutoDraw(True)
                
                # if text_3 is active this frame...
                if text_3.status == STARTED:
                    # update params
                    pass
                
                # if text_3 is stopping this frame...
                if text_3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_3.tStartRefresh + 0.8-frameTolerance:
                        # keep track of stop time/frame for later
                        text_3.tStop = t  # not accounting for scr refresh
                        text_3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_3.stopped')
                        # update status
                        text_3.status = FINISHED
                        text_3.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in show_letters_pracComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "show_letters_prac" ---
            for thisComponent in show_letters_pracComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('show_letters_prac.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.300000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 2 repeats of 'span_size_prac'
        
        
        # --- Prepare to start Routine "recall_prac" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('recall_prac.started', globalClock.getTime())
        key_resp_9.keys = []
        key_resp_9.rt = []
        _key_resp_9_allKeys = []
        instructions_6.setText('Answer and press enter.')
        # keep track of which components have finished
        recall_pracComponents = [key_resp_9, text_feedback_3, instructions_6]
        for thisComponent in recall_pracComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recall_prac" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *key_resp_9* updates
            waitOnFlip = False
            
            # if key_resp_9 is starting this frame...
            if key_resp_9.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                key_resp_9.frameNStart = frameN  # exact frame index
                key_resp_9.tStart = t  # local t and not account for scr refresh
                key_resp_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_9.started')
                # update status
                key_resp_9.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_9.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_9.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_9.getKeys(keyList=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'backspace','return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_9_allKeys.extend(theseKeys)
                if len(_key_resp_9_allKeys):
                    key_resp_9.keys = [key.name for key in _key_resp_9_allKeys]  # storing all keys
                    key_resp_9.rt = [key.rt for key in _key_resp_9_allKeys]
                    key_resp_9.duration = [key.duration for key in _key_resp_9_allKeys]
                    # was this correct?
                    if (key_resp_9.keys == str(correct_answer_3)) or (key_resp_9.keys == correct_answer_3):
                        key_resp_9.corr = 1
                    else:
                        key_resp_9.corr = 0
            
            # *text_feedback_3* updates
            
            # if text_feedback_3 is starting this frame...
            if text_feedback_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_feedback_3.frameNStart = frameN  # exact frame index
                text_feedback_3.tStart = t  # local t and not account for scr refresh
                text_feedback_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_feedback_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_feedback_3.started')
                # update status
                text_feedback_3.status = STARTED
                text_feedback_3.setAutoDraw(True)
            
            # if text_feedback_3 is active this frame...
            if text_feedback_3.status == STARTED:
                # update params
                text_feedback_3.setText(screen_text_3, log=False)
            # Run 'Each Frame' code from code_11
            if("backspace" in key_resp_9.keys):
                key_resp_9.keys.remove("backspace") 
                
                if(len(key_resp_9.keys) > 0):
                    key_resp_9.keys.pop() #remove backspace
                    
            elif("return" in key_resp_9.keys):
                key_resp_9.keys.remove("return") #remove return
                
                if(len(key_resp_9.keys) > 0):
                    screen_text_3 = ''.join(key_resp_9.keys)
                    thisExp.addData("complex_recall_prac", screen_text_3)
                
                continueRoutine = False
            
            if(len(key_resp_9.keys) > 3):
                key_resp_9.keys.pop() #prevent typing after 3 keys entered
            
            
            screen_text_3 = ''.join(key_resp_9.keys)
            screen_text_3 = screen_text_3.upper()
            
            # *instructions_6* updates
            
            # if instructions_6 is starting this frame...
            if instructions_6.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                instructions_6.frameNStart = frameN  # exact frame index
                instructions_6.tStart = t  # local t and not account for scr refresh
                instructions_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instructions_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instructions_6.started')
                # update status
                instructions_6.status = STARTED
                instructions_6.setAutoDraw(True)
            
            # if instructions_6 is active this frame...
            if instructions_6.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recall_pracComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recall_prac" ---
        for thisComponent in recall_pracComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('recall_prac.stopped', globalClock.getTime())
        # check responses
        if key_resp_9.keys in ['', [], None]:  # No response was made
            key_resp_9.keys = None
            # was no response the correct answer?!
            if str(correct_answer_3).lower() == 'none':
               key_resp_9.corr = 1;  # correct non-response
            else:
               key_resp_9.corr = 0;  # failed to respond (incorrectly)
        # store data for span_recall_prac (TrialHandler)
        span_recall_prac.addData('key_resp_9.keys',key_resp_9.keys)
        span_recall_prac.addData('key_resp_9.corr', key_resp_9.corr)
        if key_resp_9.keys != None:  # we had a response
            span_recall_prac.addData('key_resp_9.rt', key_resp_9.rt)
            span_recall_prac.addData('key_resp_9.duration', key_resp_9.duration)
        # the Routine "recall_prac" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "feedback_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('feedback_2.started', globalClock.getTime())
        # Run 'Begin Routine' code from code_12
        if (''.join(key_resp_9.keys).upper() == correct_answer_3.upper()):
            msg2 = 'Correct'
            msg2Color = 'green'
        else:
            msg2 = 'Incorrect'
            msg2Color = 'red'
        
        feedback_msg_3.setColor(msg2Color, colorSpace='rgb')
        feedback_msg_3.setText(msg2)
        # keep track of which components have finished
        feedback_2Components = [feedback_msg_3]
        for thisComponent in feedback_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_msg_3* updates
            
            # if feedback_msg_3 is starting this frame...
            if feedback_msg_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                feedback_msg_3.frameNStart = frameN  # exact frame index
                feedback_msg_3.tStart = t  # local t and not account for scr refresh
                feedback_msg_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_msg_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_msg_3.started')
                # update status
                feedback_msg_3.status = STARTED
                feedback_msg_3.setAutoDraw(True)
            
            # if feedback_msg_3 is active this frame...
            if feedback_msg_3.status == STARTED:
                # update params
                pass
            
            # if feedback_msg_3 is stopping this frame...
            if feedback_msg_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_msg_3.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_msg_3.tStop = t  # not accounting for scr refresh
                    feedback_msg_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_msg_3.stopped')
                    # update status
                    feedback_msg_3.status = FINISHED
                    feedback_msg_3.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback_2" ---
        for thisComponent in feedback_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('feedback_2.stopped', globalClock.getTime())
        # Run 'End Routine' code from code_12
        if (''.join(key_resp_9.keys).upper() == correct_answer_3.upper()):
            prac_acc = prac_acc + 1
            thisExp.addData("complex_prac_corr", "1")
        else:
            span_recall_prac.finished = False
            continueRoutine = True
            prac_acc = 0 #reset score if one trial is wrongly responded to
            thisExp.addData("complex_prac_corr", "0")
        
        if (prac_acc >= 3): 
            span_recall_prac.finished = True
            continueRoutine = False
        #stop practice if 3 consecutive trials correct
        
        random.shuffle(stimuli_letters)
        chosen_stimuli_3 = stimuli_letters[0:2] 
        correct_answer_3 = ''.join(chosen_stimuli_3) 
        loop_i_3 = 0
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 999 repeats of 'span_recall_prac'
    
    
    # --- Prepare to start Routine "instructions_complex" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructions_complex.started', globalClock.getTime())
    key_resp_10.keys = []
    key_resp_10.rt = []
    _key_resp_10_allKeys = []
    # keep track of which components have finished
    instructions_complexComponents = [instructions_7, key_resp_10]
    for thisComponent in instructions_complexComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_complex" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_7* updates
        
        # if instructions_7 is starting this frame...
        if instructions_7.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            instructions_7.frameNStart = frameN  # exact frame index
            instructions_7.tStart = t  # local t and not account for scr refresh
            instructions_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_7.started')
            # update status
            instructions_7.status = STARTED
            instructions_7.setAutoDraw(True)
        
        # if instructions_7 is active this frame...
        if instructions_7.status == STARTED:
            # update params
            pass
        
        # *key_resp_10* updates
        waitOnFlip = False
        
        # if key_resp_10 is starting this frame...
        if key_resp_10.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            key_resp_10.frameNStart = frameN  # exact frame index
            key_resp_10.tStart = t  # local t and not account for scr refresh
            key_resp_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_10, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_10.started')
            # update status
            key_resp_10.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_10.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_10.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_10.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_10.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_10_allKeys.extend(theseKeys)
            if len(_key_resp_10_allKeys):
                key_resp_10.keys = _key_resp_10_allKeys[-1].name  # just the last key pressed
                key_resp_10.rt = _key_resp_10_allKeys[-1].rt
                key_resp_10.duration = _key_resp_10_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_complexComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_complex" ---
    for thisComponent in instructions_complexComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructions_complex.stopped', globalClock.getTime())
    # check responses
    if key_resp_10.keys in ['', [], None]:  # No response was made
        key_resp_10.keys = None
    thisExp.addData('key_resp_10.keys',key_resp_10.keys)
    if key_resp_10.keys != None:  # we had a response
        thisExp.addData('key_resp_10.rt', key_resp_10.rt)
        thisExp.addData('key_resp_10.duration', key_resp_10.duration)
    thisExp.nextEntry()
    # the Routine "instructions_complex" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    span_recall = data.TrialHandler(nReps=999, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='span_recall')
    thisExp.addLoop(span_recall)  # add the loop to the experiment
    thisSpan_recall = span_recall.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisSpan_recall.rgb)
    if thisSpan_recall != None:
        for paramName in thisSpan_recall:
            globals()[paramName] = thisSpan_recall[paramName]
    
    for thisSpan_recall in span_recall:
        currentLoop = span_recall
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisSpan_recall.rgb)
        if thisSpan_recall != None:
            for paramName in thisSpan_recall:
                globals()[paramName] = thisSpan_recall[paramName]
        
        # set up handler to look after randomisation of conditions etc
        span_size = data.TrialHandler(nReps=span_count, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='span_size')
        thisExp.addLoop(span_size)  # add the loop to the experiment
        thisSpan_size = span_size.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisSpan_size.rgb)
        if thisSpan_size != None:
            for paramName in thisSpan_size:
                globals()[paramName] = thisSpan_size[paramName]
        
        for thisSpan_size in span_size:
            currentLoop = span_size
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisSpan_size.rgb)
            if thisSpan_size != None:
                for paramName in thisSpan_size:
                    globals()[paramName] = thisSpan_size[paramName]
            
            # set up handler to look after randomisation of conditions etc
            repeat_if_wrong = data.TrialHandler(nReps=999, method='random', 
                extraInfo=expInfo, originPath=-1,
                trialList=data.importConditions('../materials/processing_only_2.xlsx'),
                seed=None, name='repeat_if_wrong')
            thisExp.addLoop(repeat_if_wrong)  # add the loop to the experiment
            thisRepeat_if_wrong = repeat_if_wrong.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisRepeat_if_wrong.rgb)
            if thisRepeat_if_wrong != None:
                for paramName in thisRepeat_if_wrong:
                    globals()[paramName] = thisRepeat_if_wrong[paramName]
            
            for thisRepeat_if_wrong in repeat_if_wrong:
                currentLoop = repeat_if_wrong
                thisExp.timestampOnFlip(win, 'thisRow.t')
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        inputs=inputs, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisRepeat_if_wrong.rgb)
                if thisRepeat_if_wrong != None:
                    for paramName in thisRepeat_if_wrong:
                        globals()[paramName] = thisRepeat_if_wrong[paramName]
                
                # --- Prepare to start Routine "trials_processing_2" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('trials_processing_2.started', globalClock.getTime())
                operation_2.setText(Problem)
                key_resp_7.keys = []
                key_resp_7.rt = []
                _key_resp_7_allKeys = []
                instructions_10.setText('Answer?')
                # keep track of which components have finished
                trials_processing_2Components = [operation_2, key_resp_7, operation_feedback_3, instructions_10]
                for thisComponent in trials_processing_2Components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "trials_processing_2" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *operation_2* updates
                    
                    # if operation_2 is starting this frame...
                    if operation_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                        # keep track of start time/frame for later
                        operation_2.frameNStart = frameN  # exact frame index
                        operation_2.tStart = t  # local t and not account for scr refresh
                        operation_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(operation_2, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'operation_2.started')
                        # update status
                        operation_2.status = STARTED
                        operation_2.setAutoDraw(True)
                    
                    # if operation_2 is active this frame...
                    if operation_2.status == STARTED:
                        # update params
                        pass
                    
                    # if operation_2 is stopping this frame...
                    if operation_2.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > operation_2.tStartRefresh + RT-frameTolerance:
                            # keep track of stop time/frame for later
                            operation_2.tStop = t  # not accounting for scr refresh
                            operation_2.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'operation_2.stopped')
                            # update status
                            operation_2.status = FINISHED
                            operation_2.setAutoDraw(False)
                    
                    # *key_resp_7* updates
                    waitOnFlip = False
                    
                    # if key_resp_7 is starting this frame...
                    if key_resp_7.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_7.frameNStart = frameN  # exact frame index
                        key_resp_7.tStart = t  # local t and not account for scr refresh
                        key_resp_7.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_7, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_7.started')
                        # update status
                        key_resp_7.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(key_resp_7.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(key_resp_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if key_resp_7 is stopping this frame...
                    if key_resp_7.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_resp_7.tStartRefresh + RT-frameTolerance:
                            # keep track of stop time/frame for later
                            key_resp_7.tStop = t  # not accounting for scr refresh
                            key_resp_7.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'key_resp_7.stopped')
                            # update status
                            key_resp_7.status = FINISHED
                            key_resp_7.status = FINISHED
                    if key_resp_7.status == STARTED and not waitOnFlip:
                        theseKeys = key_resp_7.getKeys(keyList=['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_7_allKeys.extend(theseKeys)
                        if len(_key_resp_7_allKeys):
                            key_resp_7.keys = [key.name for key in _key_resp_7_allKeys]  # storing all keys
                            key_resp_7.rt = [key.rt for key in _key_resp_7_allKeys]
                            key_resp_7.duration = [key.duration for key in _key_resp_7_allKeys]
                            # was this correct?
                            if (key_resp_7.keys == str(Operation_answer)) or (key_resp_7.keys == Operation_answer):
                                key_resp_7.corr = 1
                            else:
                                key_resp_7.corr = 0
                            # a response ends the routine
                            continueRoutine = False
                    
                    # *operation_feedback_3* updates
                    
                    # if operation_feedback_3 is starting this frame...
                    if operation_feedback_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                        # keep track of start time/frame for later
                        operation_feedback_3.frameNStart = frameN  # exact frame index
                        operation_feedback_3.tStart = t  # local t and not account for scr refresh
                        operation_feedback_3.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(operation_feedback_3, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'operation_feedback_3.started')
                        # update status
                        operation_feedback_3.status = STARTED
                        operation_feedback_3.setAutoDraw(True)
                    
                    # if operation_feedback_3 is active this frame...
                    if operation_feedback_3.status == STARTED:
                        # update params
                        operation_feedback_3.setText(operation_text_3, log=False)
                    
                    # if operation_feedback_3 is stopping this frame...
                    if operation_feedback_3.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > operation_feedback_3.tStartRefresh + RT-frameTolerance:
                            # keep track of stop time/frame for later
                            operation_feedback_3.tStop = t  # not accounting for scr refresh
                            operation_feedback_3.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'operation_feedback_3.stopped')
                            # update status
                            operation_feedback_3.status = FINISHED
                            operation_feedback_3.setAutoDraw(False)
                    # Run 'Each Frame' code from code_8
                    if(len(key_resp_7.keys) > 0):
                        operation_text_3 = ''.join(key_resp_7.keys)
                        thisExp.addData("digit_answer", operation_text_3) #store response in data file after entering
                    
                        continueRoutine = False
                        
                    operation_text_3 = ''.join(key_resp_7.keys)
                    
                    # *instructions_10* updates
                    
                    # if instructions_10 is starting this frame...
                    if instructions_10.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                        # keep track of start time/frame for later
                        instructions_10.frameNStart = frameN  # exact frame index
                        instructions_10.tStart = t  # local t and not account for scr refresh
                        instructions_10.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(instructions_10, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'instructions_10.started')
                        # update status
                        instructions_10.status = STARTED
                        instructions_10.setAutoDraw(True)
                    
                    # if instructions_10 is active this frame...
                    if instructions_10.status == STARTED:
                        # update params
                        pass
                    
                    # if instructions_10 is stopping this frame...
                    if instructions_10.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > instructions_10.tStartRefresh + RT-frameTolerance:
                            # keep track of stop time/frame for later
                            instructions_10.tStop = t  # not accounting for scr refresh
                            instructions_10.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'instructions_10.stopped')
                            # update status
                            instructions_10.status = FINISHED
                            instructions_10.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, inputs=inputs, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in trials_processing_2Components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "trials_processing_2" ---
                for thisComponent in trials_processing_2Components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('trials_processing_2.stopped', globalClock.getTime())
                # check responses
                if key_resp_7.keys in ['', [], None]:  # No response was made
                    key_resp_7.keys = None
                    # was no response the correct answer?!
                    if str(Operation_answer).lower() == 'none':
                       key_resp_7.corr = 1;  # correct non-response
                    else:
                       key_resp_7.corr = 0;  # failed to respond (incorrectly)
                # store data for repeat_if_wrong (TrialHandler)
                repeat_if_wrong.addData('key_resp_7.keys',key_resp_7.keys)
                repeat_if_wrong.addData('key_resp_7.corr', key_resp_7.corr)
                if key_resp_7.keys != None:  # we had a response
                    repeat_if_wrong.addData('key_resp_7.rt', key_resp_7.rt)
                    repeat_if_wrong.addData('key_resp_7.duration', key_resp_7.duration)
                # the Routine "trials_processing_2" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "feedback_quick_2" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('feedback_quick_2.started', globalClock.getTime())
                # Run 'Begin Routine' code from code_16
                loop_o_2 += 1
                
                if not key_resp_7.keys: #if no entry
                    continueRoutine = True
                    msg4 ='Be faster! \nScore = %.1f%%' %(number_correct_3*100/loop_o_2)
                    msg4Color = 'red'
                    thisExp.addData("operation_correct", 0)
                    repeat_if_wrong.finished = False
                else:
                    if (''.join(key_resp_7.keys) == str(Operation_answer)):
                        number_correct_3 = number_correct_3 + 1
                        msg4 ='Correct! \nScore = %.1f%%' %(number_correct_3*100/loop_o_2)
                        msg4Color = 'green'
                        thisExp.addData("operation_correct", 1)
                        repeat_if_wrong.finished = True 
                    else:
                        msg4 ='Oops! That was wrong \nScore = %.1f%%' %(number_correct_3*100/loop_o_2)
                        msg4Color = 'red'
                        thisExp.addData("operation_correct", 0)
                        repeat_if_wrong.finished = False
                
                
                
                
                feedback_msg_5.setColor(msg4Color, colorSpace='rgb')
                feedback_msg_5.setText(msg4)
                operation_5.setText(Problem)
                # keep track of which components have finished
                feedback_quick_2Components = [feedback_msg_5, operation_5]
                for thisComponent in feedback_quick_2Components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "feedback_quick_2" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 0.8:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *feedback_msg_5* updates
                    
                    # if feedback_msg_5 is starting this frame...
                    if feedback_msg_5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        feedback_msg_5.frameNStart = frameN  # exact frame index
                        feedback_msg_5.tStart = t  # local t and not account for scr refresh
                        feedback_msg_5.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(feedback_msg_5, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'feedback_msg_5.started')
                        # update status
                        feedback_msg_5.status = STARTED
                        feedback_msg_5.setAutoDraw(True)
                    
                    # if feedback_msg_5 is active this frame...
                    if feedback_msg_5.status == STARTED:
                        # update params
                        pass
                    
                    # if feedback_msg_5 is stopping this frame...
                    if feedback_msg_5.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > feedback_msg_5.tStartRefresh + 0.8-frameTolerance:
                            # keep track of stop time/frame for later
                            feedback_msg_5.tStop = t  # not accounting for scr refresh
                            feedback_msg_5.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'feedback_msg_5.stopped')
                            # update status
                            feedback_msg_5.status = FINISHED
                            feedback_msg_5.setAutoDraw(False)
                    
                    # *operation_5* updates
                    
                    # if operation_5 is starting this frame...
                    if operation_5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        operation_5.frameNStart = frameN  # exact frame index
                        operation_5.tStart = t  # local t and not account for scr refresh
                        operation_5.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(operation_5, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'operation_5.started')
                        # update status
                        operation_5.status = STARTED
                        operation_5.setAutoDraw(True)
                    
                    # if operation_5 is active this frame...
                    if operation_5.status == STARTED:
                        # update params
                        pass
                    
                    # if operation_5 is stopping this frame...
                    if operation_5.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > operation_5.tStartRefresh + 0.8-frameTolerance:
                            # keep track of stop time/frame for later
                            operation_5.tStop = t  # not accounting for scr refresh
                            operation_5.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'operation_5.stopped')
                            # update status
                            operation_5.status = FINISHED
                            operation_5.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, inputs=inputs, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in feedback_quick_2Components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "feedback_quick_2" ---
                for thisComponent in feedback_quick_2Components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('feedback_quick_2.stopped', globalClock.getTime())
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-0.800000)
                thisExp.nextEntry()
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed 999 repeats of 'repeat_if_wrong'
            
            
            # --- Prepare to start Routine "show_letters_2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('show_letters_2.started', globalClock.getTime())
            text.setText(chosen_stimuli_2[loop_i_2])
            # Run 'Begin Routine' code from code_9
            loop_i_2 += 1
            # keep track of which components have finished
            show_letters_2Components = [text]
            for thisComponent in show_letters_2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "show_letters_2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text* updates
                
                # if text is starting this frame...
                if text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    text.frameNStart = frameN  # exact frame index
                    text.tStart = t  # local t and not account for scr refresh
                    text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.started')
                    # update status
                    text.status = STARTED
                    text.setAutoDraw(True)
                
                # if text is active this frame...
                if text.status == STARTED:
                    # update params
                    pass
                
                # if text is stopping this frame...
                if text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text.tStartRefresh + 0.8-frameTolerance:
                        # keep track of stop time/frame for later
                        text.tStop = t  # not accounting for scr refresh
                        text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text.stopped')
                        # update status
                        text.status = FINISHED
                        text.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in show_letters_2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "show_letters_2" ---
            for thisComponent in show_letters_2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('show_letters_2.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.300000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed span_count repeats of 'span_size'
        
        
        # --- Prepare to start Routine "recall_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('recall_2.started', globalClock.getTime())
        key_resp_8.keys = []
        key_resp_8.rt = []
        _key_resp_8_allKeys = []
        instructions_5.setText('Answer and press enter.')
        # keep track of which components have finished
        recall_2Components = [key_resp_8, text_feedback_2, instructions_5]
        for thisComponent in recall_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recall_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *key_resp_8* updates
            waitOnFlip = False
            
            # if key_resp_8 is starting this frame...
            if key_resp_8.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                key_resp_8.frameNStart = frameN  # exact frame index
                key_resp_8.tStart = t  # local t and not account for scr refresh
                key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_8.started')
                # update status
                key_resp_8.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_8.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_8.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_8.getKeys(keyList=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'backspace','return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_8_allKeys.extend(theseKeys)
                if len(_key_resp_8_allKeys):
                    key_resp_8.keys = [key.name for key in _key_resp_8_allKeys]  # storing all keys
                    key_resp_8.rt = [key.rt for key in _key_resp_8_allKeys]
                    key_resp_8.duration = [key.duration for key in _key_resp_8_allKeys]
                    # was this correct?
                    if (key_resp_8.keys == str(correct_answer_2)) or (key_resp_8.keys == correct_answer_2):
                        key_resp_8.corr = 1
                    else:
                        key_resp_8.corr = 0
            
            # *text_feedback_2* updates
            
            # if text_feedback_2 is starting this frame...
            if text_feedback_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_feedback_2.frameNStart = frameN  # exact frame index
                text_feedback_2.tStart = t  # local t and not account for scr refresh
                text_feedback_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_feedback_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_feedback_2.started')
                # update status
                text_feedback_2.status = STARTED
                text_feedback_2.setAutoDraw(True)
            
            # if text_feedback_2 is active this frame...
            if text_feedback_2.status == STARTED:
                # update params
                text_feedback_2.setText(screen_text_2, log=False)
            # Run 'Each Frame' code from code_10
            if("backspace" in key_resp_8.keys):
                key_resp_8.keys.remove("backspace") 
                
                if(len(key_resp_8.keys) > 0):
                    key_resp_8.keys.pop() #remove backspace
                    
            elif("return" in key_resp_8.keys):
                key_resp_8.keys.remove("return") #remove return
                
                if(len(key_resp_8.keys) > 0):
                    screen_text_2 = ''.join(key_resp_8.keys)
                    thisExp.addData("complex_recall", screen_text_2) #store response in data file after entering
                
                continueRoutine = False
            
            if(len(key_resp_8.keys) > 20):
                key_resp_8.keys.pop() #prevent typing after 20 keys entered
            
            
            screen_text_2 = ''.join(key_resp_8.keys)
            screen_text_2 = screen_text_2.upper()
            
            # *instructions_5* updates
            
            # if instructions_5 is starting this frame...
            if instructions_5.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                instructions_5.frameNStart = frameN  # exact frame index
                instructions_5.tStart = t  # local t and not account for scr refresh
                instructions_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instructions_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instructions_5.started')
                # update status
                instructions_5.status = STARTED
                instructions_5.setAutoDraw(True)
            
            # if instructions_5 is active this frame...
            if instructions_5.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recall_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recall_2" ---
        for thisComponent in recall_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('recall_2.stopped', globalClock.getTime())
        # check responses
        if key_resp_8.keys in ['', [], None]:  # No response was made
            key_resp_8.keys = None
            # was no response the correct answer?!
            if str(correct_answer_2).lower() == 'none':
               key_resp_8.corr = 1;  # correct non-response
            else:
               key_resp_8.corr = 0;  # failed to respond (incorrectly)
        # store data for span_recall (TrialHandler)
        span_recall.addData('key_resp_8.keys',key_resp_8.keys)
        span_recall.addData('key_resp_8.corr', key_resp_8.corr)
        if key_resp_8.keys != None:  # we had a response
            span_recall.addData('key_resp_8.rt', key_resp_8.rt)
            span_recall.addData('key_resp_8.duration', key_resp_8.duration)
        # Run 'End Routine' code from code_10
        if (''.join(key_resp_8.keys).upper() == correct_answer_2.upper()):
            span_count = span_count + 1 #increase set size
            complex_wrong = 0 #reset wrong number of repsonses
            thisExp.addData("complex_recall_corr", "1") 
            thisExp.addData("complex_recall_corrkeys", correct_answer_2.upper())
        else:
            complex_wrong = complex_wrong + 1
            thisExp.addData("complex_recall_corr", "0")
            thisExp.addData("complex_recall_corrkeys", correct_answer_2.upper())
        
        if (complex_wrong == 2):
            span_recall.finished = True
            continueRoutine = False #stop task if 2 consec wrong trials in set size
        
        
        random.shuffle(stimuli_letters)
        chosen_stimuli_2 = stimuli_letters[0:span_count] #update choose n letters
        correct_answer_2 = ''.join(chosen_stimuli_2) #update join n letters
        
        loop_i_2 = 0
        # the Routine "recall_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 999 repeats of 'span_recall'
    
    
    # --- Prepare to start Routine "finished" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('finished.started', globalClock.getTime())
    key_resp_12.keys = []
    key_resp_12.rt = []
    _key_resp_12_allKeys = []
    # keep track of which components have finished
    finishedComponents = [finish, key_resp_12]
    for thisComponent in finishedComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "finished" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *finish* updates
        
        # if finish is starting this frame...
        if finish.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            finish.frameNStart = frameN  # exact frame index
            finish.tStart = t  # local t and not account for scr refresh
            finish.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(finish, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'finish.started')
            # update status
            finish.status = STARTED
            finish.setAutoDraw(True)
        
        # if finish is active this frame...
        if finish.status == STARTED:
            # update params
            pass
        
        # *key_resp_12* updates
        waitOnFlip = False
        
        # if key_resp_12 is starting this frame...
        if key_resp_12.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            key_resp_12.frameNStart = frameN  # exact frame index
            key_resp_12.tStart = t  # local t and not account for scr refresh
            key_resp_12.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_12, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_12.started')
            # update status
            key_resp_12.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_12.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_12.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_12.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_12.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_12_allKeys.extend(theseKeys)
            if len(_key_resp_12_allKeys):
                key_resp_12.keys = _key_resp_12_allKeys[-1].name  # just the last key pressed
                key_resp_12.rt = _key_resp_12_allKeys[-1].rt
                key_resp_12.duration = _key_resp_12_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in finishedComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "finished" ---
    for thisComponent in finishedComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('finished.stopped', globalClock.getTime())
    # check responses
    if key_resp_12.keys in ['', [], None]:  # No response was made
        key_resp_12.keys = None
    thisExp.addData('key_resp_12.keys',key_resp_12.keys)
    if key_resp_12.keys != None:  # we had a response
        thisExp.addData('key_resp_12.rt', key_resp_12.rt)
        thisExp.addData('key_resp_12.duration', key_resp_12.duration)
    thisExp.nextEntry()
    # the Routine "finished" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
