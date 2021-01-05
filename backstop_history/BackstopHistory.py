################################################################################
#
#   BackstopHistory - Class used for containing and assembling  a
#                     backstop history
#
#     Author: Gregg Germain
#
#   Original: BackstopHistory.py
#
# Update: March 14, 2018
#         Gregg Germain
#         Non-Load Event Tracking (NLET)mechanism, and the ACIS Ops
#         Backstop History Assembly modules into acis_thermal_check.
#
# Update: February, 2020
#         Javier Gonzales/John Zuhone
#            - Workflow for Conda build and releases
#
# Update: June 1, 2020
#         Gregg Germain
#           - Accommodate Maneuver-Only loads
#           - Replace ParseCM and Commanded States
#           - Accomodate in-situ ECS measurments within a Normal load
#
# Update: November 4, 2020
#         Gregg Germain
#         Tom Aldcroft
#           - Accomodate shiny
#
################################################################################
from __future__ import print_function
import copy
import glob
import logging
import numpy as np
import pickle
import os
from pathlib import Path

from backstop_history import LTCTI_RTS

import kadi.commands

from cxotime import CxoTime


# -------------------------------------------------------------------------------
#
#  globfile
#
# -------------------------------------------------------------------------------
def globfile(pathglob):
    """Return the one file name matching ``pathglob``.  Zero or multiple
    matches raises an IOError exception."""

    files = glob.glob(pathglob)
    if len(files) == 0:
        raise IOError('No files matching %s' % pathglob)
    elif len(files) > 1:
        raise IOError('Multiple files matching %s' % pathglob)
    else:
        return files[0]

def config_logger(verbose):
    """
    Set up console logger.

    Parameters
    ----------
    verbose : integer
        Indicate how verbose we want the logger to be.
        (0=quiet, 1=normal, 2=debug)
    """

    # Disable auto-configuration of root logger by adding a null handler.
    # This prevents other modules (e.g. Chandra.cmd_states) from generating
    # a streamhandler by just calling logging.info(..).
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass
    rootlogger = logging.getLogger()
    rootlogger.addHandler(NullHandler())
    logger = logging.getLogger("backstop_history")
    logger.setLevel(logging.DEBUG)

    # Set numerical values for the different log levels
    loglevel = {0: logging.CRITICAL,
                1: logging.INFO,
                2: logging.DEBUG}.get(verbose, logging.INFO)

    formatter = logging.Formatter('%(message)s')

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(loglevel)
    logger.addHandler(console)

    return logger


class BackstopHistory(object):

    def __init__(self, cont_file_name='ACIS-Continuity.txt',
                 NLET_tracking_file_path='/data/acis/LoadReviews/NonLoadTrackedEvents.txt',
                 logger=None, verbose=1):
        if logger is None:
            logger = config_logger(verbose)
        self.logger = logger
        self.legal_events = ['LTCTI', 'MAN', 'TOO', 'S107', 'GO']
        self.master_list = []
        self.rev_to_take = []
        self.load_list = []
        self.backstop_list = []
        self.backstop_file_path_list = []
        self.load_type_list = []
        self.continuity_file_name = cont_file_name
        self.NLET_tracking_file_path = NLET_tracking_file_path
        self.Review_ToFC = None
        self.STOP_time = None
        self.S107_time = None
        self.TOO_ToFC = None
        self.trim_time = None
        self.power_cmd_list = ['WSPOW00000', 'WSPOW0002A', 'WSVIDALLDN']
        self.end_event_time = None  # End time to use for event searches

        # Full path to RTS files
        self.RTS = LTCTI_RTS.LTCTI_RTS(os.path.dirname(__file__))

        # Dtype definition for the ACISspecific lines in the CR* Backstop file
        self.ACIS_specific_dtype = [('event_date', 'U20'),
                                    ('event_time', '<i8'),
                                    ('cmd_type', 'U20'),
                                    ('packet_or_cmd', 'U80')]

        # Create a Dtype for the Continuity Info array
        self.cont_dtype = [('base_load', '|S20'),
                           ('cont_file', '|S80'),
                           ('load_type', '|S10'),
                           ('load_tofc', '|S25')]

        # Create the SCS-107 command set that's relevant for ACIS
        # Now the SCS-107 on board is an RTS so the times and
        # VCDU's in this data structure are bogus.  Meaningful times
        # will be entered in a copy of this data structure when the commands
        # are integrated into a backstop history
        # NOTE: Never write into this attribute. Always make a copy
        self.scs107_bs_cmds = [# --------------------- NEW SIMTRANS -----------------------------
                               {'idx': -1,
                                'date': '1900:001',
                                'type': 'SIMTRANS',
                                'tlmsid': 'None',
                                'scs': 107,
                                'step': 1,
                                'time': -1,
                                'timeline_id': 0,
                                'vcdu': 0000000,
                                'params': {'pos': -99616}},
                                # --------------------- NEW AA00000000 -----------------------------
                                {'idx': -1,
                                 'date': '1900:001',
                                 'type': 'ACISPKT',
                                 'tlmsid': 'AA00000000',
                                 'scs': 107,
                                 'step': 1,
                                 'time': -1,
                                 'timeline_id': 0,
                                 'vcdu': 0000000,
                                 'params': {'cmds': 3, 'words': 3, 'packet(40)': 'D80000300030603001300'}},

                               # --------------------- NEW AA00000000 -----------------------------
                                {'idx': -1,
                                 'date': '1900:001',
                                 'type': 'ACISPKT',
                                 'tlmsid': 'AA00000000',
                                 'scs': 107,
                                 'step': 1,
                                 'time': -1,
                                 'timeline_id': 0,
                                 'vcdu': 0000000,
                                 'params': {'cmds': 3, 'words': 3, 'packet(40)': 'D80000300030603001300'}},
                               
                               # --------------------- NEW WSPOW00000 -----------------------------
                               {'idx': -1,
                                'date': '1900:001',
                                'type': 'ACISPKT',
                                'tlmsid': 'WSPOW00000',
                                'scs': 132, 'step': 195,
                                'time': -1,
                                'timeline_id': 0,
                                'vcdu': 0000000,
                                'params': {'cmds': 5,
                                           'words': 7,
                                           'packet(40)': 'D8000070007030500200000000000010000'}} ]

        # Create the MP_TARGQUAT command that specifies a maneuver's final
        # position.
        # VCDU's, SCS, and STEP values in this data structure are bogus.  Meaningful times
        # will be entered in a copy of this data structure when the command
        # is integrated into a backstop history
        # NOTE: Never write into this attribute. Always make a copy
        # --------------------- MP_TARGQUAT ------------------------
        self.MAN_bs_cmds =  {'idx': -1,
                             'date': '1900:001',
                             'type': 'MP_TARGQUAT',
                             'tlmsid': 'AOUPTARQ',
                             'scs': 129,
                             'step': 796,
                             'time': -1,
                             'timeline_id': 0,
                             'vcdu': 0000000,
                             'params': {'cmds': 8, 'q1': 0.0, 'q2': 0.0, 'q3': 0.0, 'q4': 0.0}}

        # Create the AOMANUVR command that initiates a maneuver
        # VCDU's, SCS, and STEP values in this data structure are bogus.  Meaningful times
        # will be entered in a copy of this data structure when the command
        # is integrated into a backstop history
        # NOTE: Never write into this attribute. Always make a copy
        # --------------------- AOMANUVR ------------------------
        self.AOMANUVR_bs_cmd = {'idx': -1,
                                'date': '1900:001',
                                'type': 'COMMAND_SW',
                                'tlmsid': 'AOMANUVR',
                                'scs': 129,
                                'step': 755,
                                'time': -1,
                                'timeline_id': 0,
                                'vcdu': 0000000,
                                'params': {'hex': '8034101', 'msid': 'AOMANUVR'}}

        # Create examples of the individual power commands that might get executed by CAP
        # NOTE: Never write into these attributes. Always make a copy

        # --------------------- WSVIDALLDN -----------------------------
        # TLMSID= WSVIDALLDN, CMDS= 4, WORDS= 5, PACKET(40)= D800005000506050020000000000            , SCS= 132, STEP= 44
        self.WSVIDALLDN_bs_cmd =  {'idx': -1,
                                   'date': '1900:001',
                                   'type': 'ACISPKT',
                                   'tlmsid': 'WSVIDALLDN',
                                   'scs': 133,
                                   'step': 577,
                                   'time': -1,
                                   'timeline_id': 0,
                                   'vcdu': 0000000,
                                   'params': {'cmds': 4,
                                              'words': 5,
                                              'packet(40)': 'D800005000506050020000000000'}}

        # --------------------- WSPOW00000 -----------------------------
        # TLMSID= WSPOW00000, CMDS= 5, WORDS= 7, PACKET(40)= D8000070007030500200000000000010000     , SCS= 131, STEP= 196
        self.WSPOW00000_bs_cmd = {'idx': -1,
                                  'date': '2020:305:02:52:33.492',
                                  'type': 'ACISPKT',
                                  'tlmsid': 'WSPOW00000',
                                  'scs': 132, 'step': 195,
                                  'time': -1,
                                  'timeline_id': 0,
                                  'vcdu': 0000000,
                                  'params': {'cmds': 5,
                                             'words': 7,
                                             'packet(40)': 'D8000070007030500200000000000010000'}}

        # --------------------- WSPOW0002A -----------------------------
        # TLMSID= WSPOW0002A, CMDS= 5, WORDS= 7, PACKET(40)= D80000700073E800020000000000001002A     , SCS= 131, STEP= 170
        self.WSPOW0002A_bs_cmd = {'idx': -1,
                                  'date': '2020:304:20:09:46.677',
                                  'type': 'ACISPKT',
                                  'tlmsid': 'WSPOW0002A',
                                  'scs': 132,
                                  'step': 22,
                                  'time': -1,
                                  'timeline_id': 0,
                                  'vcdu': 0000000,
                                  'params': {'cmds': 5,
                                             'words': 7,
                                             'packet(40)': 'D80000700073E800020000000000001002A'}}

    #-------------------------------------------------------------------------------
    #
    #  get_backstop_continuity_path
    #
    #-------------------------------------------------------------------------------
    def get_continuity_file_info(self, oflsdir):
        """ Given an ofls directory, open the Continuity text file
            within the OFLS directory; read the continuity file path,
            from the first line, the type of load the *REVIEW* load is (not
            the continuity load), on the second line.  If the review load
            is not the normal weekly load (e.g. scs-107 or TOO) grab the time
            of interrupt also from the second line.

               - The load in oflsdir was built using another load as Continuity
                 assuming that the build was either a normal weekly load or a TOO
                 load. You wouldn't be using this function if this is a return to
                 science load after a shutdown of some sort.

             Tack the continuity file name (e.g. ACIS-Continuity.txt) to the path. Open up the
             Continuity text file and read the continuity load path.

             Return the  path to that continuity file ofls directory.

             INPUTS: The OFLS "review" directory

            OUTPUTS: The full path to the continuity ofls backstop file
                     The type of load the REVIEW load is.
                     The time of interrupt if the REVIEW load is an interrupt
                     load.
        """
        # `oflsdir` is of the form <root>/2018/MAY2118/ofls
        oflsdir = Path(oflsdir)

        # Require that oflsdir starts with /data/acis unless the environment
        # variable ALLOW_NONSTANDARD_OFLSDIR is set.
        allow_nonstandard_oflsdir = 'ALLOW_NONSTANDARD_OFLSDIR' in os.environ

        if (not allow_nonstandard_oflsdir
                and oflsdir.parts[:3] != ('/', 'data', 'acis')):
            raise ValueError('--backstop_file must start with /data/acis. To remove this '
                             'restriction set the environment variable '
                             'ALLOW_NONSTANDARD_OFLSDIR to any value.')

        oflsdir_root = oflsdir.parents[2]  # Supplies the <root>

        # Does a Continuity file exist for the input path
        ofls_cont_fn = oflsdir / self.continuity_file_name

        if ofls_cont_fn.exists():

            # Open the Continuity text file in the ofls directory and read the name of the
            # continuity load date (e.g. FEB2017).  then close the file
            ofls_cont_file = open(ofls_cont_fn, 'r')

            # Read the first line...the path to the continuity load. The
            # continuity path in the file is hardwired to a /data/acis path,
            # independent of user-specified `oflsdir` (e.g.
            # /data/acis/LoadReviews/2018/MAY2118/ofls), so if a non-standard
            # OFLS dir path is allowed then fix that by putting the last 3 parts
            # (2018/MAY2118/ofls) onto the oflsdir root.
            pth = Path(ofls_cont_file.readline().strip())
            if allow_nonstandard_oflsdir:
                continuity_load_path = str(Path(oflsdir_root, *pth.parts[-3:]))
            else:
                continuity_load_path = str(pth)

            # Read the entire second line - load type and possibly the interrupt time
            type_line = ofls_cont_file.readline()
            # Split the line
            split_type_line = type_line.split()

            # The review load type is always there, and always the first item on the line
            #  so capture it
            review_load_type = split_type_line[0]

            # If the review load type is not "Normal", grab the interrupt time
            # or set the interrupt time to "None"
            if review_load_type.upper() != "NORMAL":
                interrupt_time = split_type_line[1]
            else:
                interrupt_time = None

            # Done with the file...close it
            ofls_cont_file.close()

            # Return the Continuity load path to the caller.
            return continuity_load_path, review_load_type, interrupt_time
        else:
            return None, None, None

#-------------------------------------------------------------------------------
#
#   get_bs_cmds - Get the backstop commands that live in the OFLS directories.
#                 These always start with the characters "CR"
#
#-------------------------------------------------------------------------------
    def get_bs_cmds(self, oflsdir):
        """
        Given the path to an ofls directory, this method will call the "globfile"
        to obtain the name of the backstop file that represents the built load.
        It then calls kadi.commands.get_cmds_from_backstop to obtain the list of commands
        Review and Continuity loads appear in the ....ofls/ subdir and always
        begin with the characters "CR"

        INPUT: oflsdir = Path to the OFLS directory (string)

        OUTPUT   : bs_cmds = A list of the ommands within the backstop file
                             in ofls directory that represents the  built load.
                                -  list of dictionary items
        """
        backstop_file_path = globfile(os.path.join(oflsdir, 'CR*.backstop'))

        self.logger.info("GET_BS_CMDS - Using backstop file %s" % backstop_file_path)

        # append this to the list of backstop files that are processed
        self.backstop_file_path_list.append(backstop_file_path)

        # Extract the name of the backstop file from the path
        bs_name = os.path.split(backstop_file_path)[-1]

        # Read the commands located in that backstop file
        bs_cmds = kadi.commands.get_cmds_from_backstop(backstop_file_path)
        # This next line will need to be restored ONCE SHINY IS READY
        bs_cmds = bs_cmds.as_list_of_dict()

        self.logger.info("GET_BS_CMDS - Found %d backstop commands between %s and %s" % (len(bs_cmds),
                                                                                         bs_cmds[0]['date'],
                                                                                         bs_cmds[-1]['date']))

        # Return both the backstop commands in cmd states format and the
        # name of the backstop file

        return bs_cmds, bs_name

    #-------------------------------------------------------------------------------
    #
    #    get_vehicle_only_bs_cmds - Get the backstop commands that live in the
    #                               OFLS directories. These always start with the
    #                               characters "VR"
    #
    #-------------------------------------------------------------------------------
    def get_vehicle_only_bs_cmds(self, oflsdir):
        """
        Given the path to an ofls directory, this method will call the "globfile"
        to obtain the name of the backstop file that represents the built load.
        It then calls  kadi.commands.get_cmds_from_backstop to obtain the list of commands
        Vehicle_only loads appear in the ....ofls/vehicle/ subdir and always
        begin with the characters "VR"

        INPUT: oflsdir = Path to the OFLS directory (string)

        OUTPUT   : bs_cmds = A list of the ommands within the backstop file
                             in ofls directory that represents the  built load.
                                -  list of dictionary items
        """
        backstop_file_path = globfile(os.path.join(oflsdir, "vehicle", 'VR*.backstop'))
        self.logger.info('GET_BS_CMDS - Using backstop file %s' % backstop_file_path)

        # Extract the name of the backstop file from the path
        bs_name = os.path.split(backstop_file_path)[-1]

        # Read the commands located in that backstop file and convert to
        # list of dict.
        bs_cmds = kadi.commands.get_cmds_from_backstop(backstop_file_path)
        # This next line will need to be restored ONCE SHINY IS READY
        bs_cmds = bs_cmds.as_list_of_dict()

        self.logger.info('GET_VEHICLE_ONLY_CMDS - Found %d backstop commands between %s and %s' % (len(bs_cmds),
                                                                                                   bs_cmds[0]['date'],
                                                                                                   bs_cmds[-1]['date']))

        return bs_cmds, bs_name

    #-------------------------------------------------------------------------------
    #
    # CombineNormal - Combine the Continuity backstop commands with the review load
    #                 backstop commands, taking any overlap into account.
    #                 Also checks to see if there was an "in situ" ECS measurement
    #                 such as the ECS measurement that occurred in the JUL2720
    #                 NORMAL load
    #
    #                 Call this when you are reviewing a normal week-to-week load
    #
    #-------------------------------------------------------------------------------
    def CombineNormal(self, cont_bs_cmds, rev_bs_cmds):
        """
         Combine the Continuity backstop commands with the review load
         backstop commands, taking any overlap into account.

         NOTE: By "review" load we mean the load being reviewed this week OR
               a combination of one or more continuity loads with the load
               being reviewed this week.  This routine can be called multiple
               times - tacking continuity loads to the start of the "master list"

               NOTE: There can be one or more commands at the start of the
                     Review load whose time stamp is prior to the time stamp
                     of the last command in the continuity load. All commands
                     are executed because review commands are put is a different
                     SCS slot than the continuity commands.  therefore all commands
                     from both backstop files must be preserved and interleaved
                     correctly.

                 Call this when you are reviewing a normal week-to-week load

                The numpy array reurend by process RTS has this DTYPE:

                self.RTS_dtype = [('date', '|U20'),
                                  ('time','<f8'),
                                  ('statement', '|U20'),
                                  ('mnemonic', '|U20'),
                                  ('substitution_parameter',  '|U20'),
                                  ('substitution_parameter_value',  '|U20'),
                                  ('DELTA','<f8'),
                                  ('SCS_NUM', '|U5')]


                 INPUTS: Continuity load backstop file commands
                         Review Load Backstop file

                OUTPUTS: Date-sorted Backstop commands of the combined Continuity
                         and Review loads.
        """
        # First get the start and stop dates and times for the Review Load.
        # Capture the Time of First Command from the rev_bs_cmds
        Time_of_First_Command = rev_bs_cmds[0]['time']

        # Record the Review Load Time of First Command (Tofc)
        self.Review_ToFC = Time_of_First_Command

        # Next step is to set the Master List equal to the concatenation of
        # the continuity load commands and the review load commands
        # commands with no trimming, since this is a Normal load
        self.master_list = cont_bs_cmds+rev_bs_cmds
        # Sort the master list
        self.master_list = sorted(self.master_list, key=lambda k: k['time'])

        # Now scan the NLET file for any Event that occurs between the
        # start of the continuity load and the end of the review load.
        # For now we won't make this inclusive but subsequent
        # new and exciting ideas on how to operate may make that necessary.
        # So first search the NLET file for events
        event_list = self.Find_Events_Between_Dates(self.master_list[0]['time'], self.end_event_time)

        # If there are events to process.......
        if event_list != []:
            # There are, so process them all
            for eachevent in event_list:
                # split the string on spaces
                splitline = eachevent.split()
                # If the event found is a LTCTI measurement...
                if splitline[1] == 'LTCTI':
                    # .....process it.
                    # Since this is a LTCTI, process it feeding the already-split
                    # event line.
                    # Process_LTCTI appends the vent commands to self.master_list
                    self.Process_LTCTI(splitline, self.Review_ToFC)
                elif splitline[1] in self.power_cmd_list:
                    # We probably ran a CAP to execute a power command such as WSPOW0002A
                    # So insert the power command into the historical Backstop file you are building.
                    self.Process_Power_Cmd( splitline)
                else: # NOT an LTCTI nor a power command
                    print('Detected:\n    ', eachevent)

        # This is a list of dicts. Sort the list based upon the Chandra Date
        # string located in the key: "date". This will interleave all the
        # commands correctly.
        self.master_list = sorted(self.master_list, key=lambda k: k['time'])

        # Move the end event time back to the beginning of the assembled history
        self.end_event_time = self.master_list[0]['time']

        # Return the sorted command list to the caller
        return self.master_list



#-------------------------------------------------------------------------------
#
# CombineTOO - Combine the Continuity backstop commands with the review load
#              backstop commands, without overlap.
#
#                 Call this when you are reviewing a TOO load
#
#-------------------------------------------------------------------------------
    def CombineTOO(self, cont_bs_cmds, rev_bs_cmds):
        """
         Combine the Continuity backstop commands with the review load
         backstop commands, without overlap.  The Continuity load is
         cut at the time of the first command in the TOO load. If there
         are commands whose execution time is precisely the cut time, they
         will be kept and executed.  The possible AOACRSTD pre-TOFC command
         in the TOO load is placed appropriately.

         NOTE: By "review" load we mean the load being reviewed this week OR
               a combination of one or more continuity loads with the load
               being reviewed this week.  This routine can be called multiple
               times - tacking continuity loads to the start of the "master list"

                 Call this when you are reviewing a TOO load or if a
                 continuity load was broken by a TOO load.

                 INPUTS: Continuity load backstop file commands
                         Review Load Backstop file

                         NOTE: end_event_time is initialized by ATC to the end of the review load.

                OUTPUTS: Backstop commands of the combined Continuity and Review
                         loads.
        """
        # First get the start and stop dates and times for the Review Load.
        # Capture the Time of First Command from the rev_bs_cmds
        Time_of_First_Command = rev_bs_cmds[0]['time']

        # Record the TOO Load Time of First Command (ToFC)
        self.TOO_ToFC = Time_of_First_Command

        # Get all the Continuity commands up to and including the time of first command of
        # the Review load
        # NOTE: This will automatically take care of the fact that one or more of the first commands
        # in the new load will come before the end of commands in the continuity load.
        #
        self.master_list = [cmd for cmd in cont_bs_cmds if cmd['time'] <= rev_bs_cmds[0]['time']]

        # Now concatenate the review load taking all the commands
        # to the master list
        self.master_list = self.master_list + rev_bs_cmds
        # Sort the master list
        self.master_list = sorted(self.master_list, key=lambda k: k['time'])

        # MASTER  = Trimmed Continuity + Review Load.
  
        # Now scan the NLET file for any Event that occurs between the
        # start of the continuity load and the end of the review load.
        # For now we won't make this inclusive but subsequent
        # new and exciting ideas on how to operate may make that necessary.
        # So first search the NLET file for events
        event_list = self.Find_Events_Between_Dates(self.master_list[0]['time'], self.end_event_time)

        # If there are events to process.......
        if event_list != []:
            # There are, so process them all
            for eachevent in event_list:
                # split the string on spaces
                splitline = eachevent.split()
                # If the event found is a LTCTI measurement...
                if splitline[1] == 'LTCTI':
                    # ......process it feeding the routine the
                    # Review Load Time of First Command
                    self.Process_LTCTI(splitline, self.TOO_ToFC)
                elif splitline[1] in self.power_cmd_list:
                    # We probably ran a CAP to execute a power command such as WSPOW0002A
                    # So insert the power command into the historical Backstop file you are building.
                    self.Process_Power_Cmd( splitline)
                else: # Neither an LTCTI nor a power command
                    print('Detected:\n    ', eachevent)

        # Sort the master list based upon time
        self.master_list = sorted(self.master_list, key=lambda k: k['time'])

        # Move the end event time back to the beginning of the assembled history
        self.end_event_time = self.master_list[0]['time']

        return self.master_list


#-------------------------------------------------------------------------------
#
# Process_LTCTI - process the submitted LTCTIline from the NLET file
#
#-------------------------------------------------------------------------------
    def Process_LTCTI(self, ltcti_event, trim_time):
        """
            Inputs: ltcti_event - Event line from the NLET file, split on spaces, indicating
                                  a LTCTI entry

                                - format: 2020:147:02:08:00    LTCTI   1527     1_4_CTI    000:16:00:00

                      trim_date - date/time after which the continuity load has to be trimmed.
                                  This could be: NORMAL Review Load ToFC - which results in NO trimming
                                                 TOO cut time
                                                 SCS-107 or STOP time
                                - Only Continuity files get trimmed.

               NOTE: Upon entry, self.master_list contains 

            Output: None returned:  self.master_list has been updated with the LTCTI commands.

            LTCTI's can occur during shutdowns, within a Normal load (JUL2720 IRU swap), and
            across loads ( e.g. MAY2620---MAY2420).  So when processing LTCTI's the algorithm has
            to look for the first Stop Science command (AA00000000) that occurs AFTER the start of
            the LTCTI.
        """
        RTS_start_date = ltcti_event[0]
        self.RTS.CAP_num  = ltcti_event[2]
        self.RTS.RTS_name = ltcti_event[3]
        self.RTS.NUM_HOURS = ltcti_event[4]

        self.logger.info("\nLTCTI Measurement FOUND %s CAP # %s RTS file %s\n" % (RTS_start_date, str(self.RTS.CAP_num),self.RTS.RTS_name ))

        # Process the specified RTS file and get a time-stamped numpy array of the data
        ltcti_cmd_list = self.RTS.processRTS(self.RTS.RTS_name, self.RTS.SCS_NUM, self.RTS.NUM_HOURS, RTS_start_date)

        # Now convert the numpy array into SKA command format which is a list of dicts
        LTCTI_bs_cmds = self.RTS.convert_ACIS_RTS_to_ska_parse(ltcti_cmd_list)

        # We need to find the first ACISPKT command in the review load that
        # comes after the start of the LTCTI first command, and is ALSO
        # a Stop Science ('AA00000000').
        # IMPORTANT: The LTCTI run may have started in the Continuity load
        #            but it will end either because it runs to completion with
        #            it's own Stop Science commands, OR a Stop Sciecne in the review load
        # To do that we obtain the start time of the timed LTCTI command set
        ltcti_cmd_list_start_time = CxoTime(RTS_start_date).secs

        # Initialize the ACIS_specific_cmds as an empty array of DTYPE self.ACISPKT_dtype
        ACIS_specific_cmds = np.array( [], dtype = self.ACIS_specific_dtype)

        # Next, collect all commands in the review backstop file which
        # are between the LTCTI start time and the LTCTI stop time, inclusive
        #
        # Start by getting a copy of all the backstop file commands, directly
        # from the official backstop file, which are pertinent to ACIS and
        # this operation: ACISPKT, ORBPOINTS etc.
        #
        # Work your way backwards through self.backstop_file_path_list, except
        # the first file which is the review load. Trim off any commands that
        # occur after the STOP time.processing
        #
        for eachCR_file in self.backstop_file_path_list[:0:-1]:
            # Read all the Continuity commands pertinent to ACIS filtered by get_ACIS_backstop_cmds
            all_file_cmds = self.get_ACIS_backstop_cmds(eachCR_file)
            # Trim the commands that occur on or after the trim date
            filter_arr = all_file_cmds['event_time'] <= trim_time
            ACIS_specific_cmds = np.append(ACIS_specific_cmds, all_file_cmds[filter_arr], axis=0)

        # At this point you have all the Continuity pertinent commands assembled.
        # Now get the pertinent commands coming from the Review Load
        all_file_cmds = self.get_ACIS_backstop_cmds(self.backstop_file_path_list[0])
        #....and append them
        ACIS_specific_cmds = np.append(ACIS_specific_cmds, all_file_cmds, axis = 0)

        # Now find all review backstop commands which are a Stop Science and
        # which occurs AFTER the start of the LTCTI run.
        cut_cmd_indices = np.where( (ACIS_specific_cmds[:]['event_time'] >= ltcti_cmd_list_start_time) & \
                                                (ACIS_specific_cmds[:]['packet_or_cmd'] == 'AA00000000') )

        # A maneuver-only load may NOT have any Stop Science commands in it.
        # For example the May2620 Maneuver-Only load
        # So check to see if you got a result for cut_cmd_indices.
        # if you did then use it to determine whether or not you need to trim the
        # LTCTI commands.  If not, then the LTCI ran to completion
        # If no stop sciences in the (maneruver-only) review load...
        if cut_cmd_indices[0].size == 0:
            print(' NO STOP SCIENCES IN REVIEW LOAD')
            #...then the KTCTI runs to completion
            trimmed_LTCTI_bs_cmds = LTCTI_bs_cmds
        else: # Else there was a stop science found in the review load...
            # Get the first AA00 from the REVIEW load
            bs_load_stop_science = ACIS_specific_cmds[cut_cmd_indices[0][0]]

            # There was a stop science in the review load so trim any
            # LTCTI CLD commands that occurred ON or AFTER the
            # Return to Science Time of First Command.
            # If the first AA00 in the Review load comes AFTER the AA00 in the LTCTI
            # Command list, the LTCTI AA00 is preserved.
            trimmed_LTCTI_bs_cmds = self.Trim_bs_cmds_After_Date(bs_load_stop_science['event_time'], LTCTI_bs_cmds)

        # Append the LTCTI commands to the Master list
        # They are out of time order but will be sorted in the calling routine.
        self.master_list += trimmed_LTCTI_bs_cmds

#-------------------------------------------------------------------------------
#
# Process_MAN - process the submitted maneuver line from the NLET file
#
#-------------------------------------------------------------------------------
    def Process_MAN(self, man_event):
        """
            Inputs: man_event - Event line from the NLET file, split on spaces, indicating
                                a maneuver
        """
        MAN_date = man_event[0]
        pitch = man_event[2]
        roll = man_event[3]
        q1 = man_event[4]
        q2 = man_event[5]
        q3 = man_event[6]
        q4 = man_event[7]

        # If this is a legal maneuver, process it
        if pitch != 0.0:

            self.logger.info("NEW MANEUVER FOUND %s pitch: %s" % (MAN_date, str(pitch)))

            # Now form and add the command (in SKA.Parse format - i.e. dict) which
            # specifies the MP_TARGQUAT
            new_maneuver = copy.deepcopy(self.MAN_bs_cmds)

            # Set the dates, times and Q's
            new_maneuver['date'] = MAN_date
            new_maneuver['params']['q1'] = float(q1)
            new_maneuver['params']['q2'] = float(q2)
            new_maneuver['params']['q3'] = float(q3)
            new_maneuver['params']['q4'] = float(q4)
            new_maneuver['time'] = CxoTime(MAN_date).secs

            # Tack the maneuver to the Master List
            self.master_list.append(new_maneuver)

        else: # It's a bogus maneuver entry - the user didn't specify good Q's
            self.logger.warning("Bogus Maneuver Entry! Quaternions badly specified: \n"
                                "Bad Q's: %g, %g, %g, %g " % (q1, q2, q3, q4) +
                                "...therefore bogus pitch and roll: %g, %g" % (pitch, roll))

        # Create the AOMANUVR command and add it to the Master List.
        # This command actually kicks the maneuver off.
        # NOTE: This is done whether or not the maneuver data was good.
        #       It allows the loop to continue looking for maneuvers
        aoman = copy.deepcopy(self.AOMANUVR_bs_cmd)
        aoman_time = CxoTime(MAN_date).secs + 1
        aoman_date = CxoTime(aoman_time).date
        aoman['date'] = aoman_date
        aoman['time'] = aoman_time
        # Tack the maneuver to the Master List
        # Tacking the command to the Master list doesn't really do much if there
        # is no AOUPTARQ command. But it allows you to search for subsequent maneuver
        # commands

        # Append the commands for this maneuver to the Master List
        self.master_list.append(aoman)

    #-------------------------------------------------------------------------------
    #
    # Process_MAN - process the submitted maneuver line from the NLET file
    #
    #-------------------------------------------------------------------------------
    def Process_Power_Cmd(self, power_cmd_event):
        """
            Inputs: power_cmd_event - Event line from the NLET file, split on spaces, indicating
                                      which power command was executed and at what time

                        Example:
                            #       Time        Event         CAP num
                            #-------------------------------------------------------------------------------
                            2020:238:02:48:00    WSPOW0002A   1540

            NOTE: If you get to this method, the command has already been checked for existence. So
                  you can rest assured that the necessary data structurs exist
        """
        # Log that you see it
        self.logger.info("\nNEW Power Command Found %s power cmd: %s" % (CxoTime(power_cmd_event[0]).date, power_cmd_event[1]) )
        # Process the power command. Make a copy of the appropriate power command attribute
        if power_cmd_event[1] == 'WSVIDALLDN':
            new_pwr_cmd = copy.deepcopy(self.WSVIDALLDN_bs_cmd)
        elif power_cmd_event[1] == 'WSPOW00000':
            new_pwr_cmd = copy.deepcopy(self.WSPOW00000_bs_cmd)
        else:  # It can only be a POW2a if you get here
            new_pwr_cmd = copy.deepcopy(self.WSPOW0002A_bs_cmd)

	# Next insert the date and tie which comes from the NLET line
        new_pwr_cmd['date'] = CxoTime(power_cmd_event[0]).date
        new_pwr_cmd['time'] = CxoTime(power_cmd_event[0]).secs

        # Now append this power command to the Master List.
        self.master_list.append(new_pwr_cmd)


#-------------------------------------------------------------------------------
#
# CombineSTOP - Combine the Continuity backstop commands with the review load
#               backstop commands, when both the Vehicle and Science loads have
#               been stopped.
#
#-------------------------------------------------------------------------------
    def CombineSTOP(self, cont_bs_cmds, rev_bs_cmds, shutdown_date):

        # Convert and record shutdown date to seconds:
        self.STOP_time = CxoTime(shutdown_date).secs

        # Trim the Continuity commands list to include only those
        # commands whose excution time is before the shutdown time
        self.master_list = [cmd for cmd in cont_bs_cmds if (cmd['time'] < self.STOP_time) ]

        #
        # IMPORTANT: At this point, self.master_list should consist ONLY of
        #            the continuity load TRIMMED to the STOP time

        # When the Spacecraft does a Full Stop, an SCS-107 is excuted.
        # So make a copy of the SCS-107 commands and populate the times.  These
        # are: SIMTRANS, AA00, AA00, WSPOW. Then concatenate the 107 commands
        # to the master list
        scs107_bs_cmds = copy.deepcopy(self.scs107_bs_cmds)

        # The starting time for the first scs107 command will be at the stop time

        # 1 second after the last
        # command in the TRIMMED master list
        base_time = self.STOP_time

        # populate the date and time slots of each command incrementing the times
        # by one second
        for eachcmd in scs107_bs_cmds:
            eachcmd['time'] = base_time
            eachcmd['date'] = CxoTime(base_time).date

            # Increment the base time by one second
            base_time += 1.0

        # Now concatenate scs107 commands specific to this event to the master list
        # which has been trimmed to include only those commands that executed prior to
        # the shutdown. No need to sort these at this point
        self.master_list += scs107_bs_cmds

        # MASTER LIST = Trimmed Continuity + SCS-107 commands

        # Now we need to process any events that appear in the NonLoadEventTracker.txt
        # file whose times are after the continuity load start time, but before the subsequent
        # STOP time of the review load. These events include:
        #
        #     "MAN" includes:
        #           NSM - pitch change to 90 degrees: ALL Stop
        #           BSH - stuck at some pitch - ALL Stop
        #           OCC Pitch Maneuver - Move to a new pitch.
        #
        #     Long Term CTI events (LTCTI)
        #
        # So first search the NLET file for events between the start of the Continuity load
        # and the end of the Review Load.
        event_list = self.Find_Events_Between_Dates(self.master_list[0]['time'], self.end_event_time)

        # If there are events to process.......
        if event_list != []:
            # There are, so process them all
            for eachevent in event_list:
                # split the string on spaces
                splitline = eachevent.split()
                # If this is a MANEUVER, event, process it and add it to the Master List
                if splitline[1] == 'MAN':
                    self.Process_MAN(splitline)
                elif splitline[1] == 'LTCTI':
                    # Since this is a LTCTI, process it feeding the routine the
                    # Review Load Time of First Command
                    self.Process_LTCTI(splitline, self.STOP_time)
                elif splitline[1] in self.power_cmd_list:
                    # We probably ran a CAP to execute a power command such as WSPOW0002A
                    # So insert the power command into the historical Backstop file you are building.
                    self.Process_Power_Cmd( splitline)
                else:
                    print('Detected:\n    ', eachevent)

        # MASTER LIST = Trimmed Continuity +
        #               SCS-107 commands +
        #               Any Maneuvers and/or LTCTI's that occurred

        # Finally,  concatenate the review load tacking all the commands
        # to the master list.
        # NOTE: In subsequent calls to this method the TOFC of rev_bs_cmds will be
        #       the start of the *assembled* load history. Only in the case of the first
        #       call is the TOFC of rev_bs_cmds also the TOFC of the actual Review Load.
        newlist =  self.master_list + rev_bs_cmds

        # MASTER LIST = Trimmed Continuity +
        #               SCS-107 commands +
        #               Any Maneuvers and/or LTCTI's that occurred +
        #               Review Load

        # sort the master list based on time so tht events occur at the correct
        # moment of time.
        self.master_list = sorted(newlist, key=lambda k: k['time'])

        # Move the end event time back to the beginning of the assembled history
        self.end_event_time = self.master_list[0]['time']

        # Now return the sorted master list which contains
        return self.master_list

#-------------------------------------------------------------------------------
#
# Combine107 - Combine the Continuity backstop commands with the review load
#              backstop commands, without overlap and without including any
#              command after the specified SCS-107 time
#
#                 Call this when you are reviewing an SCS-107 load
#
#-------------------------------------------------------------------------------
    def Combine107(self, cont_bs_cmds, vo_bs_cmds, rev_bs_cmds, shutdown_date):
        """
         Combine the Continuity backstop commands with the review load
         backstop commands, without overlap IN THE SCIENCE LOAD and without including any
         SCIENCE command after the specified SCS-107 time.  The Continuity load is
         cut at the time of shutdown.

         THe SCS 107 commands are inserted after the time of shutdown

         LTCTI is looked for in the NLET file and if it exists, it is inserted
         into the history

         There is a check for the  AOACRSTD command in the TOO load and it's
         placed accordingly via a sort.

         Steps:

             1) Trim all commands from the continuity load on and after the
                SCS-107 interrupt time.

             2) Add in the SCS-107 Shutdown commands (SIMTRANS, AA, AA, WSPOW)

             3) Check for any LTCTI's. If they exist, add them to the master list.

             4) Trim the beginning of the VO load to eliminate that part of the
                continuity load you are keeping.

             5) Trim the back end of the VR file to the Review load start time

             6)  Concatenate the VO remnant to the assembled continuity.

             7) Concatenate the Review Load

         NOTE: By "review" load we mean the load being reviewed this week OR
               a combination of one or more continuity loads with the load
               being reviewed this week.  This routine can be called multiple
               times - tacking continuity loads to the start of the "master list"

                 Call this when you are reviewing a normal week-to-week load

                 INPUTS: Continuity load backstop file commands
                         Review Load Backstop file
                         Time of Shutdown

                OUTPUTS: Backstop commands of the combined Continuity and Review
                         loads.
        """
        # Convert shutdown date to seconds:
        self.S107_time = CxoTime(shutdown_date).secs

        # STEP 1

        # Trim the Continuity commands list to include only those commands whose excution
        # time is less than the time of first command of the Review load (or assembled bs list)
        self.master_list = [cmd for cmd in cont_bs_cmds if (cmd['time'] < self.S107_time)]

        # MASTER LIST = Trimmed Continuity

        # Capture the end of the master list (trimmed continuity )
        # so that you can use it later for the Verhicle Only Load Cut. You want it to
        # be right after the SCS-107 time. But we are adding in any LTCTI runs first.
        # That would move the end time of master list too far. The same thing would happen
        # to the LTCTI if we swapped them in time.
        vo_cut_date = self.master_list[-1]['date']

        # STEP 2 SCS-107 SIMTRANS AND STOP

        # Now make a copy of the SCS-107 commands and populate the times. These
        # are: SIMTRANS, AA00, AA00, WSPOW. Then concatenate the 107 commands
        # to the master list
        scs107_bs_cmds = copy.deepcopy(self.scs107_bs_cmds)
        # The starting time for the first scs107 command will be 1 second after the last
        # command in the TRIMMED master list
        base_time = CxoTime(self.S107_time).secs + 1

        # populate the date and time slots of each command incrementing the times by one second
        for eachcmd in scs107_bs_cmds:
            eachcmd['time'] = base_time
            eachcmd['date'] = CxoTime(base_time).date
            # Increment the base time by one second
            base_time += 1.0

        # Now concatenate scs107 commands specific to this event to the master list
        # which has been trimmed to include only those commands that executed prior to
        # the shutdown. No need to sort these at this point, because the base time was
        # the last time of the trimmed continuity load plus 1 second.
        self.master_list += scs107_bs_cmds

        # MASTER LIST = Trimmed Continuity + SCS-107 COMMANDS

        # STEP 3
        self.logger.info("STEP - Process any and all events between: %s - %s" % (self.master_list[-1]['date'],rev_bs_cmds[0]['date']))

        # Now we need to process any events that appear in the NonLoadEventTracker.txt
        # file whose times are after the stop time, but before the subsequent
        # start science of the review load. The only events non-load events
        # that canoccur are Long Term ECS measurements
        #
        # So first search the NLET file for any LTECS events between the start
        # of the Continuity load and the end of the REview Load
        event_list = self.Find_Events_Between_Dates(self.master_list[0]['time'], self.end_event_time)

        # If there are events to process.......
        if event_list != []:
            # There are, so process them all
            for eachevent in event_list:
                # split the string on spaces
                splitline = eachevent.split()
                # If the event found is a LTCTI measurement...
                if splitline[1] == 'LTCTI':
                    # .....process it feeding the routine the
                    # Review Load Time of First Command
                    self.Process_LTCTI(splitline, self.S107_time)
                elif splitline[1] in self.power_cmd_list:
                    # We probably ran a CAP to execute a power command such as WSPOW0002A
                    # So insert the power command into the historical Backstop file you are building.
                    self.Process_Power_Cmd( splitline)
                else:
                    print('Detected:\n    ', eachevent)


        # MASTER LIST = Trimmed Continuity + SCS-107 COMMANDS + ANY LTCTI's + ANY INDIVIDUAL POWER COMMANDS.


        # Trim the Vehicle Only command list by removing vo commands occuring
        # prior to the stop date.
        vo_bs_cmds_trimmed = self.Trim_bs_cmds_Before_Date(vo_cut_date, vo_bs_cmds)

        # STEP 5

        # Trim all commands in the VO list that occur AFTER the first command
        # of the return to science load
        vo_bs_cmds_trimmed = self.Trim_bs_cmds_After_Date(rev_bs_cmds[0]['time'], vo_bs_cmds_trimmed)

        # STEP 6

        # Concatenate the trimmed VO list
        self.master_list += vo_bs_cmds_trimmed

        # MASTER LIST = Trimmed Continuity +
        #                 SCS-107 COMMANDS +
        #                      ANY LTCTI's +
        #                   ANY POWER CMDS +
        #                VEHICLE ONLY CMDS

        # Finally,  concatenate the review load tacking all the commands
        # to the master list.
        # NOTE: In subsequent calls to this method the TOFC of rev_bs_cmds will be
        #       the start of the *assembled* load history. Only in the case of the first
        #       call is the TOFC of rev_bs_cmds also the TOFC of the actual Review Load.
        newlist =  self.master_list + rev_bs_cmds


        # MASTER LIST = Trimmed Continuity +
        #                 SCS-107 COMMANDS +
        #                      ANY LTCTI's +
        #                VEHICLE ONLY CMDS +
        #                      Review Load

        # sort the master list based on time so that events occur at the correct
        # moment of time.
        self.master_list = sorted(newlist, key=lambda k: k['time'])

        # Move the end event time back to the beginning of the assembled history
        self.end_event_time = self.master_list[0]['time']

        # Now return the sorted master list which contains
        return self.master_list


    #-------------------------------------------------------------------------------
    #
    # WriteCombinedCommands - Write the combined commands to a file
    #
    #-------------------------------------------------------------------------------
    def WriteCombinedCommands(self, cmd_list, outfile_path, comment = ''):
        """
        This method will write the command list out into a file whose path is specified
        in outfile_path.  Whether or not this is an original command list or a combined
        one it immaterial.

            INPUTS: command list
                    full output file specification

           OUTPUTS: Nothing returned; file written.
        """
        # Open up the file for writing
        combofile = open(outfile_path, "w")

        # Now output pertinent info from the command list.
        for eachcmd in cmd_list:
            if eachcmd['cmd'] != 'GET_PITCH':
                cmd_line = eachcmd['date'] + ' | '+ eachcmd['cmd']+ ' | '+ eachcmd['paramstr']
                combofile.write(cmd_line+'\n')

        # Output the comment whatever it is
        combofile.write('\nComment: '+comment+'\n')
        # Done with the file; close it.
        combofile.close()


    #-------------------------------------------------------------------------------
    #
    # Trim_bs_commands_After_Date - Given a list of backstop commands, remove any command
    #                               prior to the specified time
    #
    #                 INPUTS: Chandra date or time (either is ok)
    #                         List of backstop commands.
    #
    #                OUTPUTS: Trimmed list of backstop commands
    #-------------------------------------------------------------------------------
    def Trim_bs_cmds_After_Date(self, trim_date, cmd_list):
        """
        Given a list of backstop commands, remove any command ON or *AFTER* the specified time

        This can be confusing - you want to throw away all those commands that occur
        ON or AFTER the specified time. You want to keep everything that occurs BEFORE the
        cutoff time.

        INPUTS: Chandra date or time (either is ok)

                List of backstop commands.

        OUTPUTS: Trimmed list of backstop commands

        NOTE: John Scott assures me that they try to schedule the first
              command of the new load to be either between loads or in between two
              commands in the old load.  Also, they will not lengthen an observation
              by having the stop science come much later than it did in the broken load.
              They tried that once and there were myriad complaints.
        """
        # Convert the date into seconds. If it was already seconds this will not hurt.
        trim_time = CxoTime(trim_date).secs

        # Now trim the list by including only those backstop commands that occur after the time.
        trimmed_list = [cmd for cmd in cmd_list if cmd['time'] < trim_time]

        # Return the trimmed list
        return trimmed_list



    #-------------------------------------------------------------------------------
    #
    # Trim_bs_cmds_Before_Date - Given a list of backstop commands, remove any command
    #                            prior to the specified time
    #
    #                 INPUTS: Chandra date or time (either is ok)
    #                         List of backstop commands.
    #
    #                OUTPUTS: Trimmed list of backstop commands
    #-------------------------------------------------------------------------------
    def Trim_bs_cmds_Before_Date(self, trim_date, cmd_list):
        """
        Given a list of backstop commands, remove any command *PRIOR* to the specified time

        This can be confusing - you want to throw away all those commands that occur BEFORE
        the specified time. You want to keep everything that occurs ON or AFTER the
        cutoff time.

        INPUTS: Chandra date or time (either is ok)
                List of backstop commands.

        OUTPUTS: Trimmed list of backstop commands

        NOTE: John Scott assures me that you want to do a "less than" comparison
              on times and not less than or equal. Also they try to schedule the first
              command of the new load to be either between loads or in between two
              commands in the old load.  Also, they will not lengthen an observation
              by having the stop science come much later than it did in the broken load.
              They tried that once and there were myriad complaints.
        """
        # Convert the date into seconds. If it was already seconds this will not hurt.
        trim_time = CxoTime(trim_date).secs

        # Now trim the list by KEEPING only those backstop commands that occur AFTER the time.
        trimmed_list = [cmd for cmd in cmd_list if cmd['time'] >= trim_time]

        # Return the trimmed list
        return trimmed_list


    #-------------------------------------------------------------------------------
    #
    # Backchain - Given a base directory, a starting load (base_load), and a
    #             chain length, this method will successively backtrack through the
    #             Continuity text files of each load starting with the
    #             base load, and record the continuity information in a numpy array
    #
    #                 INPUTS: base_dir  (e.g. '/data/acis/LoadReviews/2017/')
    #                         base_load (e.g. 'AUG3017')
    #                         chain_length - the number of backtracks you want
    #                         to make.
    #
    #                OUTPUTS: Array of records for each back chain through the
    #                         Continuity files.
    #
    #    VITALLY IMPORTANT!!!!!!! The January 30, 2017 load was the FIRST LOAD
    #                             to have the Continuity text file stored.
    #                             Therefore you cannont back Chain further beyond
    #                             The January 30th load.
    #
    #-------------------------------------------------------------------------------
    def BackChain(self, base_load_dir, chain_length):
        """
        Given a full base load directory, and a
        chain length, this method will successively backtrack through the
        Continuity text files of each load starting with the
        base load, and record the continuity information in a numpy array

        INPUTS: base_dir  (e.g. '/data/acis/LoadReviews/2017/AUG3017/ofls')

                              chain_length - the number of backtracks you want
                             to make.

        OUTPUTS: Array of records for each back chain through the
                 Continuity files.

                 Example, for inputs:
                                     base_dir = '/data/acis/LoadReviews/2017/AUG3017/ofls'
                                     chain_length = 4

                                      The output would be:
     ('AUG3017', '/data/acis/LoadReviews/2017/AUG2817/oflsb', 'TOO', '2017:242:23:35:01.28')
     ('AUG2817', '/data/acis/LoadReviews/2017/AUG2517/oflsc', 'Normal', 'None')
     ('AUG2517', '/data/acis/LoadReviews/2017/AUG1917/oflsa', 'TOO', '2017:237:03:30:01.28')
     ('AUG1917', '/data/acis/LoadReviews/2017/AUG1417/oflsb', 'TOO', '2017:231:15:42:18.91')

        VITALLY IMPORTANT!!!!!!! The January 30, 2017 load was the FIRST LOAD
                                 to have the Continuity text file stored.
                                 Therefore you cannont back Chain further beyond
                                 The January 30th load.
        """
        # Create an empty load chain array
        load_chain = np.array([], dtype=self.cont_dtype)

        # Extract the base load week from the full path
        # This is entered in the first column of the resultant array
        base_load_week = os.path.split(base_load_dir)[-2]
        # Extract the continuity info from the Load week.
        # REMEMBER: this information is with regard to the
        # input load. It's the Continuity file that leads to the
        # Review/input; whether the review/input load is Normal, TOO
        # or SCS-107; and if the latter two - what the Time of
        # First Command is.
        continuity_info = self.get_continuity_file_info(base_load_dir)

        # What we want to do is keep tacking continuity info onto the array
        # until we get the number of entries asked for OR we run into a load
        # week that does not have a continuity file. In the latter case we
        # want to exit gracefully and give the user what we have (if anything).
        #
        # You've done one fetch. It's either loaded with continuity info
        # (if ACIS-Continuity.txt exists in the directory) or 3 None's were
        # returned.
        #
        # If you have not exceeded the requested chain length and
        # there is continuity info, tack it onto the load chain array
        while len(load_chain) < chain_length and continuity_info[0] is not None:

            continuity_load_path = continuity_info[0]

            # Load up an entry in the array for this load
            load_chain = np.r_[load_chain,
                                np.array( [ (base_load_week,
                                          continuity_load_path,
                                          continuity_info[1],
                                          continuity_info[2]) ],
                                dtype = self.cont_dtype) ]

            # Try to do it again

            # Extract the base load week from the returned full path
            # This is entered in the first column of the resultant array
            base_load_week = continuity_load_path.split('/')[-2]

            # Get the continuity info for this new week
            continuity_info = self.get_continuity_file_info(continuity_load_path)

        # Return the array of back chains
        return load_chain


    #-------------------------------------------------------------------------------
    #
    # Find_Events_Between_Dates - Given a path to a Non Load Event Tracking file,
    #                             a start time and a stop time, search the Tracking
    #                             file for any NLET event that
    #                             occurred between the start and stop times.
    #
    #-------------------------------------------------------------------------------
    def Find_Events_Between_Dates(self, tstart, tstop):
        """
        Given a path to a Non Load Event Tracking file, a start time
        and a stop time, search the Tracking file for any event that
        occurred between the start and stop times.

        What you want to use for Start and Stop times are the SCS-107
        times for tstart and the time of first command for the replan load

        The path to the Non Load Event Tracking file (NLET) is a constructor argument
        so that users can have their own version of the file. However the
        format of the file is fixed and this method expects a certain format.
        """
        # Initialize and empty Event List
        event_list = []
        # Convert the input tstart and tstop to seconds - this allows the
        # user to input either seconds or DOY format - whichever is more
        # convenient.
        tstart = CxoTime(tstart).secs
        tstop = CxoTime(tstop).secs
        # The Non Load Event Tracking file is an input so that different
        # users of this module can have different NLET files.
        nletfile = open(self.NLET_tracking_file_path, 'r')

        # Get the first line
        nletline = nletfile.readline()

        # Process each line. If it starts with a # then ignore it - it's a
        # comment
        # 
        # for as long as you have input lines......
        while nletline:

            # Check to see if it's a comment line
            if nletline[0] != '#':

                # Not a comment. So it has to be either an event:
                # e.g. LCTI, TOO, MAN STOP, S107
                # or a "GO" - which for now is ignored
                # or a blank line which ought not be there
                # If it's an event, append the string to the list
                #
                # Split the line
                splitline = nletline.split()
                if (splitline[0] != 'GO') and \
                   (CxoTime(splitline[0]).secs > tstart) and \
                   (CxoTime(splitline[0]).secs < tstop):

                    # We have found an event. append it to the list while
                    # removing the \n at the end of the string
                    event_list.append(nletline[:-1])

            # read the next line
            nletline = nletfile.readline()

        # You've read all the lines. Close the file.
        nletfile.close()

        # Return items from any found netline; or Nones if
        # no LTCTI line matched the requirements.
        return event_list


    #-------------------------------------------------------------------------------
    #
    # get_ACIS_backstop_cmds - Given a list of CR*.backstop files Read the files
    #                          and extract out commands important to ACIS
    #                          Return a data struct of the pertinent commands in
    #                          time order.
    #
    #-------------------------------------------------------------------------------
    def get_ACIS_backstop_cmds(self, infile):
        """
        This method extracts command lines of interest to ACIS Ops from the
        Backstop files in the infile list

        The only input is a list of paths to one or  more backstop files.

        Backstop files are found in the ACIS ofls directory and always start
        with the letters "CR" and end with the extension ".backstop"


        At the present time, the backstop commands of interest to ACIS are:
                    All ACISPKT commands
              Perigee Passage indicators: 'OORMPDS', 'EEF1000', 'EPERIGEE', 'XEF1000', 'OORMPEN'
          SCS clear and disable commands: 'CODISAS1', 'COCLRS1'

        More can be added later if required

	The output data structure that is returned is a numpy array of 4 items:

            Event Date (DOY string)
            Event Time (seconds)
            Event Type (strings including ACISPKT, COMMAND_SW, and ORBPOINT)
            The Packet or command

        Example array entries:
           ('2020:213:01:00:03.00', 712544472, 'COMMAND_SW', 'OORMPDS'),
           ('2020:213:10:04:03.00', 712577112, 'COMMAND_SW', 'OORMPDS'),
           ('2020:213:10:04:59.00', 712577168, 'COMMAND_SW', 'OORMPEN'),
           ('2020:213:10:07:00.00', 712577289, 'ACISPKT', 'AA00000000'),
           ('2020:213:10:07:03.00', 712577292, 'ACISPKT', 'AA00000000'),
           ('2020:213:10:07:33.00', 712577322, 'COMMAND_SW', 'CODISASX'),
           ('2020:213:10:07:34.00', 712577323, 'COMMAND_SW', 'COCLRSX'),
        """
        # Create the empty array using the self.ACISPKT_dtype
        ACIS_specific_cmds = np.array( [], dtype = self.ACIS_specific_dtype)

        # These are the perigee passage indicators we want to recognize
        cmd_indicators = ['ACISPKT', 'OORMPDS', 'EEF1000', 'EPERIGEE', 'XEF1000', 'OORMPEN', 'CODISAS1', 'COCLRS1']

        # Open the file
        bsdf = open(infile, 'r')

        # Read eachline in the file and check to see if it's one we want
        # to save
        for eachline in bsdf:

        # Check if the line is one of the perigee Passage indicators
            if [True for cmd_ind in cmd_indicators if (cmd_ind in eachline)]:
                # You have stumbled upon a backstop command of interest
                # Now extract the date and TLMSID values
                # Start by splitting the line on vertical bars
                split_line = eachline.split('|')

                # Extract and clean up the date entry - remove any spaces
                packet_time = split_line[0].strip()

                # Extract the command type (e.g. 'ACISPKT' 'COMMAND_SW', 'ORBPOINT')
                cmd_type = split_line[2].strip()

                # Now split the 4th element of splitline - the "TLMSID" 
                # section - on commas, 
                # grab the first element in the split list (e.g. TLMSID= RS_0000001)
                # and split THAT on spaces
                # and take the last item which is the command packet of interest (e.g. RS_0000001)
                cmd = split_line[3].split(',')[0].split()[-1]

                # Load up an array line.  You need only grab the date, calculate
                #  the time in seconds, insert the command type, and the mnemonic
                ACIS_specific_cmds = np.r_[ACIS_specific_cmds,
                                             np.array( [ ( packet_time,
                                                           CxoTime(packet_time).secs,
                                                           cmd_type,
                                                           cmd) ],
                                                       dtype = self.ACIS_specific_dtype) ]

        # Finished reading and processing the file
        bsdf.close()

        # Return the backstop command array
        return ACIS_specific_cmds



    #-------------------------------------------------------------------------------
    #
    # write_back_chain_to_pickle - Given a backchain, write the contents of the
    #                              backchain to a pickle file. If this is ACIS format
    #                              the full path to the OFLS directory is written out.
    #
    #-------------------------------------------------------------------------------
    def write_back_chain_to_pickle(self, inst_format='ACIS',
                                   file_path='/home/gregg/DPAMODEL/dpa_check/Chain.p', chain=None):
        """
        Given a backchain, and a file path, write the contents of the backchain to a pickle file.
        If the "inst_format" argument is "ACIS' then the full path to the ACIS ofls directory is
        written out. If the inst_format is anything except ACIS, then the load week 
        (e.g. JAN0917) is extracted from the second column in the backchain array and only that is
        written out.

        The latter option allows users with their own backstop tarball directory structure to
        utilize the backchain and obtain backstop command files.

        The inst_format of the backchain must be a numpy array whose DTYPE is self.cont_dtype

        Note that the results of Backchain can be an enpty array. If so, the resultant pickle
        file will be empty.
        """
        # If the inst_format is ACIS then write out the array as is - meaning that the
        # 'cont_file' column will specify a full path such as:
        #         '/data/acis/LoadReviews/2017/OCT0917/ofls'
        if inst_format == 'ACIS':
            with open(file_path, 'w') as outfile:
                chain.dump(outfile)
            outfile.close()
        else: # Just write out the continuity week name from the second column (e.g. OCT0917)
            # Create an empty general array of detype self.cont_dtype
            gen_chain = np.array( [], dtype = self.cont_dtype)
            
            # Run through the chain, modify the second column, and append it to the general
            # array
            for eachload in chain:
                # Extract just the load week from the full continuity path
                continuity_week = eachload[1].split('/')[-2]

                # Append the  entry in the array for this load
                gen_chain = np.r_[gen_chain,
                                  np.array( [ (eachload[0],      # Base load
                                               continuity_week,  # Continuity load name
                                               eachload[2],      # Base load type
                                               eachload[3]) ],   # Interrupt time or None
                                           dtype = self.cont_dtype)  ]
            # Now you've created the generalized load array so dump that
            with open(file_path, 'w') as outfile:
                gen_chain.dump(outfile)
            outfile.close()

    #-------------------------------------------------------------------------------
    #
    # write_back_chain_to_txt - Given a backchain, write the contents of the
    #                           backchain to a text file. If this is ACIS inst_format
    #                           the full path to the OFLS directory is written out.
    #
    #-------------------------------------------------------------------------------
    def write_back_chain_to_txt(self, inst_format='ACIS',
                                file_path='/home/gregg/DPAMODEL/dpa_check/Chain.txt', chain=None):
        """
        Given a backchain, and a file path, write the contents of the backchain to a text file.
        If the "inst_format" argumen is "ACIS' then the full path to the ACIS ofls directory is
        written out. If the inst_format is anything except ACIS, then the load week 
        (e.g. JAN0917) is extracted from the second column in the backchain array and only that is
        written out.

        The latter option allows users with their own backstop tarball directory structure to
        utilize the backchain and obtain backstop command files.

        The inst_format of the backchain must be a numpy array whose DTYPE is self.cont_dtype

        Note that the results of Backchain can be an enpty array. If so, the resultant pickle
        file will be empty.
        """
        # If the inst_format is ACIS then write out the array as is - meaning that the
        # 'cont_file' column will specify a full path such as:
        #         '/data/acis/LoadReviews/2017/OCT0917/ofls'
        if inst_format == 'ACIS':
            np.savetxt(file_path, chain, fmt = '%s', newline = '\n')

        else: # Just write out the continuity week name from the second column (e.g. OCT0917)
            # Create an empty general array of detype self.cont_dtype
            gen_chain = np.array( [], dtype = self.cont_dtype)
            
            # Run through the chain, modify the second column, and append it to the general
            # array
            for eachload in chain:
                # Extract just the load week from the full continuity path
                continuity_week = eachload[1].split('/')[-2]

                # Append the  entry in the array for this load
                gen_chain = np.r_[gen_chain,
                                  np.array( [ (eachload[0],      # Base load
                                               continuity_week,  # Continuity load name
                                               eachload[2],      # Base load type
                                               eachload[3]) ],   # Interrupt time or None
                                           dtype = self.cont_dtype)  ]
            # Now you've created the generalized load array so dump that
            np.savetxt(file_path, gen_chain, fmt = '%s', newline = '\n')

    #-------------------------------------------------------------------------------
    #
    # read_back_chain_from_txt - Given a path to a txt file written by
    #                            write_back_chain_to_txt, read the contents of the
    #                            text file and store it in the specified array
    #                            variable.
    #
    #-------------------------------------------------------------------------------
    def read_back_chain_from_txt(self, file_path='/home/gregg/DPAMODEL/dpa_check/Chain.txt'):
        """
        Given a path to a txt file written by write_back_chain_to_txt,
        read the contents of the text file and store it in an array
        using DTYPE: self.cont_dtype
        """
        # Read the file
        chain = np.loadtxt(file_path, self.cont_dtype)
        # Return the array
        return chain

    #-------------------------------------------------------------------------------
    #
    # method set_backstop_lists - Given the name of a weekly load (e.g. MAR2717A)
    #                             and the name of the backstop file for that load,
    #                             insert the load name into the beginning of the class
    #                             attribute list:  self.load_list and the backstop file
    #                             name at the beginning of the self.backstop_list
    #
    #-------------------------------------------------------------------------------
    def set_backstop_lists(self, load_week, backstop_name, load_type):
        """
        Given the name of a weekly load (e.g. MAR2717A)
        and the name of the backstop file for that load,
        insert the load name into the beginning of the class
        attribute list:  self.load_list and the backstop file
        name at the beginning of the self.backstop_list

        The idea here is to maintain a history of the files and directories
        you used to create the set of backstop commands.

        After being used once, the lists should be cleared by calling
        self.clear_backstop_lists if you are running two or more histories
        in one program
        """
        if load_week is not None:
            self.load_list.insert(0, load_week)

        if backstop_name is not None:
            self.backstop_list.insert(0, backstop_name)

        if load_type is not None:
            self.load_type_list.insert(0, load_type)

    #-------------------------------------------------------------------------------
    #
    # method clear_backstop_lists - Clear out the load and backstop file history lists
    #
    #-------------------------------------------------------------------------------
    def clear_backstop_lists(self):
        """
        Clearing out  self.load_list and self.backstop_list
        """
        del self.load_list[:]
        del self.backstop_list[:]
        del self.load_type_list[:]

    #-------------------------------------------------------------------------------
    #
    # method print_backstop_lists - Print the load week, load type and backstop file
    #                               name  history lists
    #
    #-------------------------------------------------------------------------------
    def print_backstop_lists(self):
        """
        Print out self.load_list and self.backstop_list
        """
        print(self.load_list)
        print(self.backstop_list)
        print(self.load_type_list)

