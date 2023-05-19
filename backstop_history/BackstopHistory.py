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
#
# Update: June 2021
#         Gregg Germain
#           - Modify SCS-107 commands for WSPOW0002A
#           - Move history assembly functions (e.g. the big loop) 
#             out of ATC and into BSC
#           - New power commands were added to account for recently
#             added power commands/SI modes
#           - Use CR backstop files directly
#           - Create model-capable output CR*.backstop.hist backstop file
#             of the assembled history
#           - Eliminate methods not used
#           - Consolidated template command definitions into one file for easy update
#           - Eliminated files
#           - Eliminated any classes that didn't have to be classes
#           - Created local logger
#           - Logger verbosity level transmitted from model invocation command line
#
# Update: November 24, 2021
#         Gregg Germain
#         - Modifications which handle the case where a load was reviewed as a full
#           load, meaning both science and vehicle SCS slots will be activated, or
#           the case where, after the review of a "full" load, only the vehicle SCS's
#           were actually activated.  This is done by changes to LR which create
#           additional ACIS-Continuity.txt load types.
#           These new load types tell Assemble_History to read the CR*.backstop
#           or VR*.backstop file in the continuity directory
#
# Update: August 16, 2022
#               Gregg Germain
#               V4.3
#               - Added method to extract commands, based on tokens,  from
#                 the master list and create a new array
#               - Substituted 1_ECS4.RTS for 1_4_CTI.RTS
#               - Fixed SCS-107 SIMTRANS location
#               - Fixed comment typos
#
# Update: May 17, 2023
#               Gregg Germain
#               - V4.4
#               - Added 1_ECS2, 1_ECS3A, 1_ECS3B, 1_ECS4ALT, RTS files.
#               - Replaced the  the 6 chip RTS file with 1_ECS6
#
################################################################################
from __future__ import print_function

from astropy.io import ascii
from astropy.table import Table, vstack 
from astropy.time import Time

import copy
import glob
import json
import logging
import numpy as np
import pickle
import os
from pathlib import Path
import sys

from backstop_history import LTCTI_RTS

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

    # MODS NEEDED - Now create a localized handler
    logger = logging.getLogger("backstop_history")

    # Create handler which sends output to stdout
    bh_handler = logging.StreamHandler(stream = sys.stdout)

    # Set numerical values for the different log levels
    loglevel = {0: logging.CRITICAL,
                        1: logging.INFO,
                        2: logging.DEBUG}.get(verbose, logging.DEBUG)

    # Set the logging level at the LOGGER instance
    logger.setLevel(loglevel)

    # Create format...
    formatter = logging.Formatter('%(name)-3s: [%(levelname)-9s] %(message)s')
    # ... and add it to handlers
    bh_handler.setFormatter(formatter)

    # Add the updated backstop history handler to the logger
    logger.addHandler(bh_handler)

    return logger


class Backstop_History_Class(object):

    def __init__(self, cont_file_name='ACIS-Continuity.txt',
                 NLET_tracking_file_path='/data/acis/LoadReviews/NonLoadTrackedEvents.txt',
                 outdir = None,
                 verbose=2):

        logger = config_logger(verbose)

        self.logger = logger

        self.logger.debug('Backstop History V4.4: LOGGER ************************* BSHC Init' )

        self.outdir = outdir

        # Initialize variables holding review load information.
        self.review_file_name = None
        self.review_file_tstart = None
        self.review_file_tstop = None

        # This is the master list which will contain the Assembled Command History
        self.master_list = []
        self.master_ToFC = None
 
        self.backstop_file_dir = None
        self.backstop_file_list = []
        self.backstop_file_path_list = []

        self.continuity_file_name = cont_file_name

        self.NLET_tracking_file_path = NLET_tracking_file_path
        
        self.STOP_time = None
        self.S107_time = None
        self.TOO_ToFC = None
        self.trim_time = None
        self.end_event_time = None  # End time to use for event searches

        # Path to various data files such as command sequences
        self.cmd_seq_file_path = os.path.dirname(__file__)

        # Collection of specialized command sequences such as the sequence 
        # the sequence performed for an SCS-107

        self.raw_scs107_cmd_list = [ 'SIMTRANS', 'AA00000000', 'AA00000000', 'WSPOW0002A']
        self.raw_man_cmd_list = ['MP_TARGQUAT', 'AOMANUVR']
        self.raw_ltcti_cmd_list = []
        self.power_cmd_list = ['WSPOW00000', 'WSPOW0002A', 'WSVIDALLDN']

        # Attribute specifying the full path to where the assembled history file is
        # written out
        self.assembled_hist_file_path = None

        # CR Command Array DTYPE
        self.CR_DTYPE = [('commands', '|U3000'), ('time', '<f8'), ('date', 'U21')]

        # Dtype definition for the ACISspecific lines in the CR* Backstop file
        self.ACIS_specific_dtype = [('event_date', 'U20'),
                                    ('event_time', '<i8'),
                                    ('cmd_type', 'U20'),
                                    ('packet_or_cmd', 'U80')]

        # Create a Dtype for the Continuity Info array
        self.cont_dtype = [('base_load', '|U20'),
                           ('load_type', '|U10'),
                           ('load_tofc', '|U25'),
                           ('cont_file', '|U80')]

        # Read in the CR_Commands.dat file which contains the commands we may have to
        # process (e.g. power commands start/stop science etc).  Turn it into a
        # dictionary
        cmd_def_file_path = os.path.join(self.cmd_seq_file_path, 'CR_Commands.dat')
        with open(cmd_def_file_path) as f:
            CR_cmd_data = f.read()

        # reconstructing the data as a dictionary
        self.CR_cmds = json.loads(CR_cmd_data)


    #-------------------------------------------------------------------------------
    #
    #  get_backstop_continuity_path
    #
    #-------------------------------------------------------------------------------
    def get_continuity_file_info(self, oflsdir):
        """ Given an ofls directory, which should be the Continuity file ofls directory,
            tack the continuity file name (e.g. ACIS-Continuity.txt) to the path. Open up the
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
            if (review_load_type.upper() != 'NORMAL') and \
               (review_load_type.upper() != 'VO_NORMAL'):
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
    #  read_CR_backstop_file  - Given a full path to a CR*.backstop file, read in,
    #                           process, and return the commands
    #
    #-------------------------------------------------------------------------------
    def read_CR_backstop_file(self, backstop_file_path):
        with open(backstop_file_path) as f:
            cr_cmds = f.read().splitlines()

        # Calculate the times and the dates
        cr_times = [round(Time(eachdate.split()[0], format = 'yday', scale = 'utc').cxcsec,1) for eachdate in cr_cmds]
        cr_dates = [eachdate.split()[0] for eachdate in cr_cmds]

        # Now make a numpy array with two columns: cmd strings and times
        cr_cmds = np.array(list(zip(cr_cmds, cr_times, cr_dates)) , dtype = self.CR_DTYPE) 

        # Return the array created from the backstop file
        return cr_cmds


    #-------------------------------------------------------------------------------
    #                   
    # get_CR_bs_cmds - Get the backstop commands that live in the OFLS directories.
    #                  These always start with the characters "CR"
    #
    #-------------------------------------------------------------------------------
    def get_CR_bs_cmds(self, oflsdir):
        """
        Given the path to an ofls directory, this method will call the "globfile"
        to obtain the name of the CR backstop file that represents the built load.
        
        Review and Continuity loads appear in the ....ofls/ subdir and always
        begin with the characters "CR"

        INPUT: oflsdir = Path to the OFLS directory (string)

        OUTPUT   : bs_cmds = A list of the ommands within the backstop file
                             in ofls directory that represents the  built load.
                                -  list of dictionary items

        NOTE: This is a GENERAL reader. It will only return the data. 
              More specifically it will not store nor append the commands
              into the master list.

        """
        backstop_file_path = globfile(os.path.join(oflsdir, 'CR*.backstop'))

        self.logger.debug("GET_CR_BS_CMDS - Using backstop file %s" % backstop_file_path)

        # append this to the list of backstop files that are processed
        self.backstop_file_path_list.append(backstop_file_path)
 
        # Extract the name of the backstop file from the path
        bs_name = os.path.split(backstop_file_path)[-1]

        # Save the CR file name in the backstop file list
        self.backstop_file_list.append(bs_name)

        # Read and process the CR*.backstop file
        bs_cmds = self.read_CR_backstop_file(backstop_file_path)

        self.logger.info('GET_CR_BS_CMDS - Found %d backstop commands between %s and %s' % (len(bs_cmds),
                                                                                            Time(bs_cmds[0]['time'], format = 'cxcsec', scale = 'utc').yday,
                                                                                            Time(bs_cmds[-1]['time'], format = 'cxcsec', scale = 'utc').yday))
        # Return both the backstop commands and the name of the backstop file
        return bs_cmds, bs_name


    #-------------------------------------------------------------------------------
    #
    #  Read_Review_load - Method which assumes the directory you gave it contains
    #                     the Review Load Backstop file. It reads the file and
    #                     saves the commands to  the Master list
    #
    #-------------------------------------------------------------------------------
    def Read_Review_Load(self, oflsdir):
        """
        Read the backstop file located in the directory provided and store
        the commands into the master_list attribute

        Return the master list for debug use
        """
        # Read the commands
        self.master_list, self.review_file_name = self.get_CR_bs_cmds(oflsdir)

        # Save the times of the first and last command
        self.review_file_tstart = self.master_list[0]['time']
        self.review_file_tstop = self.master_list[-1]['time']

        # Set the end time for event searching to the end of the review load
        self.end_event_time = self.review_file_tstop

        # Return the master list
        return self.master_list

    #-------------------------------------------------------------------------------
    #
    #  get_vehicle_only_bs_cmds - Get the backstop commands that live in the
    #                             OFLS directories. These always start with the
    #                             characters "VR"
    #
    #-------------------------------------------------------------------------------
    def get_VR_bs_cmds(self, ofls_dir):
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
        backstop_file_path = globfile(os.path.join(ofls_dir, "vehicle", 'VR*.backstop'))
        self.logger.debug('GET_VO_BS_CMDS - Using backstop file %s' % backstop_file_path)

        # Extract the name of the backstop file from the path
        bs_name= os.path.split(backstop_file_path)[-1]

        # Read and process the CR*.backstop file
        bs_cmds = self.read_CR_backstop_file(backstop_file_path)

        self.logger.info('GET_VO_BS_CMDS - Found %d backstop commands between %s and %s' % (len(bs_cmds),
                                                                                            Time(bs_cmds[0]['time'], format = 'cxcsec', scale = 'utc').yday,
                                                                                            Time(bs_cmds[-1]['time'], format = 'cxcsec', scale = 'utc').yday))
        return bs_cmds, bs_name

#-------------------------------------------------------------------------------
#
# assemble_history - Given the load directory, and a time, assemble the
#                    history starting with the review load - which was assumed
#                    to already be loaded in self.master_list - and back chaining
#                    until the earliest command time in the assembled load 
#                    is on or before the stop time
#
#-------------------------------------------------------------------------------
    def Assemble_History(self, ofls_dir, tbegin, interrupt=False):
        """
            This method assembles a history of commands starting with the 
            load under review and extending back in time to on or before
            tbegin.

            The method assumes nothing has been read in, but that ofls_dir
            points to the directory of the load under review.

            The method first reads in the review load and stores it in the
            attribute "master_list".  Every other command set (continuity or events)
            is added to the master list.

            Inputs: ofls_dir - Path to the Review Load Directory
                               e.g. /data/acis/LoadReviews/2021/JUN2821/ofls/

                      tbegin - Time, in yday, which stops the backchaining of
                               continuity loads. 
                                  - Typically the TOFC of the Continuity load

            Outputs: The finished Master list: self.master_list
        
        """
        self.logger.debug('--------------------ASSEMBLING HISTORY. BACKSTOP LIST IS: %s' %  (self.backstop_file_list))

        # Convert tbegin to cxcsec if necessary; have both DOY string and
        # cxcsec available.
        if isinstance(tbegin, str):

            tbegin_time = Time(tbegin, format = 'yday', scale = 'utc').cxcsec
        else:
            tbegin_time =tbegin
            tbegin = Time(tbegin_time, format = 'cxcsec', scale = 'utc').yday

        self.logger.debug('\nTBEGIN_DATE IS: %s, TEBEGIN TIME IS: %s' % (tbegin, tbegin_time))

        # Save the interrupt flag and the OFLS directory into class attributes
        self.interrupt = interrupt
        self.backstop_file_dir = ofls_dir

        # Capture the TOFC of the review load
        self.master_ToFC = round(self.master_list[0]['time'], 1)

        # Capture the path to the ofls directory. The reason you do this is because 
        # as you backchain, the ofls directory you will be looking at changes. But
        # we want to retain the original load-under-review directory path in 
        # self.backstop_file and so not overwrite it. Therefore copy it to a 
        # perishable variable
        present_ofls_dir = copy.copy(self.backstop_file_dir)

        # WHILE
        # The big while loop that backchains through previous loads and concatenates the
        # proper load sections and possibly events (e.,g. maneuvers)to the Master List
        while self.master_ToFC > tbegin_time:
            
            # Obtain the Continuity information of the present ofls directory
            cont_load_path, present_load_type, scs107_date = self.get_continuity_file_info(present_ofls_dir)

            self.logger.debug('    Cont load path: %s, load type: %s, scs107 date: %s' % (cont_load_path, present_load_type, scs107_date))


            # Read the commands from the Continuity file
            # The value of present_load_type indicates whether to read the CR*.backstop file
            # or the VR*.backstop file.
            
            # Indicators of load types which require reading the VR*.backstop file
            self.vehicle_types = ['VO_NORMAL', 'VO_TOO', 'VO_SCS-107', 'VO_STOP']
            
            if present_load_type.upper() in self.vehicle_types:
                cont_cr_cmds, cont_file_name = self.get_VR_bs_cmds(cont_load_path)
            else:
                cont_cr_cmds, cont_file_name = self.get_CR_bs_cmds(cont_load_path)


            #---------------------- NORMAL ----------------------------------------
            # If the PRESENT (i.e. NOT Continuity) load type is "normal" then grab 
            # the continuity command set and concatenate those commands to the start of 
            # the Master List.
            if (present_load_type.upper() == 'NORMAL') or\
               (present_load_type.upper() == 'VO_NORMAL'):

                self.logger.debug('Processing %s: %s' % (present_load_type, self.master_list[-1]['date']))

                # Next step is to set the Master List equal to the concatenation of
                # the continuity load commands and the review load commands
                # commands with no trimming, since this is a Normal load
                self.master_list = np.append(cont_cr_cmds, self.master_list, axis=0)  

                # Sort the master list based upon the time column
                self.master_list.sort(order='time')

                # Adjust the Master List Time of First Command (ToFC)
                self.master_ToFC = round(self.master_list[0]['time'], 1)

                # Now point the "present" ofls directory to the Continuity directory
                # This instigates the back chaining.
                present_ofls_dir = cont_load_path

 
            #---------------------- TOO ----------------------------------------
            # If the PRESENT (i.e. NOT Continuity) load type is "TOO" then grab 
            # the continuity command set, trim the command set to exclude those
            # Commands discarded by the TOO cut, concatenate the remaining 
            # continuity commands to the start of the Master List.
            elif (present_load_type.upper() == 'TOO') or \
                 (present_load_type.upper() == 'VO_TOO'):
                self.logger.debug('Processing %s  at date: %s' % (present_load_type, scs107_date))

                # Convert the TOO cut time found in the ACIS-Continuity.txt file to
                # cxcsec
                too_cut_time = round(Time(scs107_date, format = 'yday', scale = 'utc').cxcsec, 1)

                # Trim off any continuity command which occurs on or after the
                # TOO cut time
                cont_cr_cmds = self.Trim_bs_cmds_After_Time(too_cut_time, cont_cr_cmds)

                # Combine the trimmed continuity commands with the master list
                self.master_list = np.append(cont_cr_cmds, self.master_list, axis=0)  

                # Sort the master list based upon the time column
                self.master_list.sort(order='time')

                # Adjust the Master List Time of First Command (ToFC)
                self.master_ToFC = round(self.master_list[0]['time'], 1)

                # Now point the operative ofls directory to the Continuity directory
                present_ofls_dir = cont_load_path


            #---------------------- STOP ----------------------------------------
            # If the PRESENT (i.e. NOT Continuity) load type is "STOP" then:
            #    1) trim the continuity command set to include only those continuity
            #      commands prior to the Cut
            #    2) Add on the SCS-107 commands that were executed at the Cut time
            #    3) Look for and insert any events (power commands, maneuvers, LTCTI's)
            #    4) append all these commands to the start of the Master List.
            elif (present_load_type.upper() == 'STOP') or \
                 (present_load_type.upper() == 'VO_STOP'):

                self.logger.debug('Processing %s at date: %s' % (present_load_type, scs107_date))

                # Convert the SCS-107 cut time found in the ACIS-Continuity.txt file to
                # cxcsec
                scs107_time = round(Time(scs107_date, format = 'yday', scale = 'utc').cxcsec, 1)

                # Trim off any continuity command which occurs on or after the
                # SCS-107 cut time
                cont_cr_cmds = self.Trim_bs_cmds_After_Time(scs107_time, cont_cr_cmds)

                # Time stamp the raw SCS-107 commands using the STOP_date
                processed_scs107_cmds = self.Process_Cmds(self.raw_scs107_cmd_list, scs107_date)

                # Append the processed SCS-107 commands to the continuity 
                # command list
                cont_cr_cmds = np.append(processed_scs107_cmds, cont_cr_cmds, axis=0)

                # Sort the continuity list based upon the time column
                cont_cr_cmds.sort(order='time')

                # Continuity = Trimmed Continuity plus time stamped SCS-107 commands

                # Combine the trimmed continuity commands with the master list
                self.master_list = np.append(cont_cr_cmds, self.master_list, axis=0)  

                # Sort the master list based upon the time column
                self.master_list.sort(order='time')

                # Adjust the Master List Time of First Command (ToFC)
                self.master_ToFC = round(self.master_list[0]['time'], 1)

                # Now point the operative ofls directory to the Continuity directory
                present_ofls_dir = cont_load_path

  #---------------------- SCS-107 ----------------------------------------
            # The load type is "SCS-107". master_list contains the review load
            # The Continuity load was read in at the top of the loop
            # So perform these functions for the 107:
            #     1) Get an updated set of SCS-107 commands
            #         - Concat to master_list and sort
            #     2) Trim the Continuity Commands removing all those after the SCS-107 time
            #         - Concat to master_list and sort
            #     3) Obtain the Vehicle Only (VO) commands
            #         - Trim those before the SCS-107 time and any after the Review Load start time
            #         - Concat to master_list and sort
            # 
            elif (present_load_type.upper() == 'SCS-107') or \
                 (present_load_type.upper() == 'VO_SCS-107'):
                # Inform coder that you are processing an SCS-107
                self.logger.debug('Processing %s  at date: %s' % (present_load_type, scs107_date))

                # Convert the TOO cut time found in the ACIS-Continuity.txt file to
                # cxcsec
                scs107_time = round(Time(scs107_date, format = 'yday', scale = 'utc').cxcsec, 1)

                # Trim off any continuity command which occurs on or after the
                # SCS-107 cut time
                cont_cr_cmds = self.Trim_bs_cmds_After_Time(scs107_time, cont_cr_cmds)

                # Time stamp the raw SCS-107 commands using the STOP_date
                processed_scs107_cmds = self.Process_Cmds(self.raw_scs107_cmd_list, scs107_date)

                # Append the processed SCS-107 commands to the continuity 
                # command list. This includes the WSPOW0002A
                cont_cr_cmds = np.append(processed_scs107_cmds, cont_cr_cmds, axis=0)

                # Sort the continuity list based upon the time column
                cont_cr_cmds.sort(order='time')

                # Continuity = Trimmed Continuity + time stamped SCS-107 commands

                # Combine the trimmed continuity commands with the master list
                self.master_list = np.append(cont_cr_cmds, self.master_list, axis=0)  

                # Master = Continuity + SCS-107 cmds

                # Sort the master list based upon the time column
                self.master_list.sort(order='time')

                # Obtain the CONTINUITY load Vehicle-Only file

                vo_cr_cmds, vo_cr_name = self.get_VR_bs_cmds(cont_load_path)

                self.logger.debug('    Got the VO load from: %s %s'% (cont_load_path, vo_cr_name))
                self.logger.debug('    VO START:  %s' % (vo_cr_cmds[0]['date']))
                self.logger.debug('     VO STOP: %s' % (vo_cr_cmds[-1]['date']))

                # Trim off all VO commands prior to scs-107 because they already ran
                vo_cr_cmds = self.Trim_bs_cmds_Before_Time(scs107_time, vo_cr_cmds)

                # Trim off all VO commands AFTER the start of the master_ list
                # because they will not run
                vo_cr_cmds = self.Trim_bs_cmds_After_Time(self.master_ToFC, vo_cr_cmds)

                self.logger.debug('    After VO TRIM: %s %s' % (scs107_date, self.master_ToFC))
                self.logger.debug('    VO START: %s' % (vo_cr_cmds[0]['date']))
                self.logger.debug('     VO STOP: %s' % (vo_cr_cmds[-1]['date']))

                # Combine the trimmed VO commands with the master list
                self.master_list = np.append(vo_cr_cmds, self.master_list, axis=0)  

                # Sort the master list based upon the time column
                self.master_list.sort(order='time')

                # Master = Continuity + SCS-107 cmds + trimmed VO cmds


                # Adjust the Master List Time of First Command (ToFC)
                self.master_ToFC = round(self.master_list[0]['time'], 1)

                # Now point the "present" ofls directory to the Continuity directory
                # This instigates the back chaining.
                present_ofls_dir = cont_load_path


            # At this point you have processed the continuity load, a VO load if required,
            # and an SCS107 command set if required and appended all that
            # to the master list. 

            # self.master_ToFC is the time of the start of the assembled history section
            # self.end_event_time points to the last time of the Review load

            #  Continuity start          Continuity end  Review start         Review end
            #  |                                  |      |                             |
            # self.master_ToFC                                            self.end_event_time

            # or, if you've been backchaining, the time of th4e first command of the
            # previous backchain.

            #  New Continuity start                     Old C-start       Old-Review end
            #  |                                        |                        
            #  self.master_ToFC                 self.end_event_time

            # The task now is to look for any NLET events that occurred between the
            # present values of self.master_ToFC and self.end_event_time

            # Now scan the NLET file for any Event that occurs between the
            # start of the newly updated master_list and the previously recorded
            # end_event_time value. You are tacking them on as you go and using
            # 
            # So search the NLET file for events....
            event_list = self.Find_Events_Between_Dates(self.master_ToFC, self.end_event_time)

            # ....and if there are events to process.......
            if len(event_list) > 0:

                # There are, so process them all
                for eachevent in event_list:
                    # split the string on spaces
                    splitline = eachevent.split()

                    # LTCTI event - Process and add to the master list
                    if splitline[1] == 'LTCTI':
                        # Process_LTCTI appends the event commands to self.master_list
                        ltcti_cmds = self.Process_LTCTI(eachevent, self.master_ToFC)
                        # Append the processed LTCTI commands
                        self.master_list = np.append(ltcti_cmds, self.master_list, axis=0)  
                        # Sort the resultant master list based upon the time column
                        self.master_list.sort(order='time')

                    # MANEUVER, event, process it and add it to the Master List
                    elif splitline[1] == 'MAN':
                        processed_man_cmds = self.Process_MAN(eachevent)

                        # Append the processed maneuver commands
                        self.master_list = np.append(processed_man_cmds, self.master_list, axis=0)  

                        # Sort the master list based upon the time column
                        self.master_list.sort(order='time')

                    # POWER COMMAND
                    elif splitline[1] in self.power_cmd_list:
                        # The SCS-107 now executes the power command WSPOW0002A
                        # So insert any OTHER  power command into the historical Backstop file you are building.
                        power_cmds = self.Process_Power_Cmd(eachevent)

                        # Append the processed POWER commands
                        self.master_list = np.append(power_cmds, self.master_list, axis=0)  

                        # Sort the resultant master list - DO NOT REVERSE
                        self.master_list.sort(order='time')

                    else: # NOT an LTCTI nor a Maneuver nor a power command
                        self.logger.info('NLET Processing, Event Detected: %s    ' % (eachevent))

            # Whether or not there were events tacked onto this assembly,
            # move the self.end_event_time to the START of the assembly.
            # If and when you append another continuity load, you will then 
            # check for events within the newly added section of the assembly
            self.end_event_time = self.master_ToFC

        # The history has been assembled. Write it out to a
        # file in the OFLS directory whose naming convention is:
        #   CR*.backstop.hist
        filespec = os.path.join(self.outdir, self.review_file_name+'.hist')

        self.logger.info('Writing assembled history to: %s' % (filespec) )

        # Set the History File Path attribute
        self.assembled_hist_file_path = filespec

        self.Write_Commands_to_File(self.master_list, filespec, 'w', comment = None)

        # Return the final ASSEMBLED history which is contained in Master List
        return self.master_list

#-------------------------------------------------------------------------------
#
# Process_MAN - process the submitted maneuver line from the NLET file
#
#-------------------------------------------------------------------------------
    def Process_MAN(self, nlet_event):
        """
            Inputs: man_event - Event line from the NLET file, split on spaces, indicating
                                a maneuver
        """
        # Split, on spaces, the maneuver event line read from the NLET Tracking file
        man_event_list = nlet_event.split()

        # Extract pertinent data to meaningful names
        MAN_date = man_event_list[0]
        pitch = man_event_list[2]
        roll = man_event_list[3]
        q1 = man_event_list[4]
        q2 = man_event_list[5]
        q3 = man_event_list[6]
        q4 = man_event_list[7]

        # If this is a legal maneuver, process it.
        # Other LR code (history-files.pl) explicitly sets the pitch to 0.0 in
        # the NLET file if the user did not or could not specify the 4
        # quaternions. The 4 Quaternion entries are set to a bogus value as 
        # well. Since we know the pitch ought not get below 45, and 0.0 is 
        # not possible then explicitly setting will work. It is converted 
        # from a string value - not a calculation.
        if pitch != 0.0:

            self.logger.info("NEW MANEUVER FOUND %s pitch: %s" % (MAN_date, str(pitch)))

            # Create the time stamped maneuver command set using MAN_date as the start time
            processed_maneuver_cmds = self.Process_Cmds(self.raw_man_cmd_list, MAN_date)

            # Now insert the Quaternion values into the MP_TARGQUAT params section.
            #     The MP_TARGQUAT is the first of the two commands
            # make a deep copy of the params section
            x = processed_maneuver_cmds[0]['commands']

            splitx = x.split('-0')
            scs_step = x.split('00000004')[-1] 

            splitx[1] = str(q1) + ',' + splitx[1].split(',')[1]
            splitx[2] = str(q2) + ',' + splitx[2].split(',')[1]
            splitx[3] = str(q3) + ',' + splitx[3].split(',')[1]
            splitx[4] = str(q4) + ',' + splitx[4].split(',')[1] + scs_step

            processed_maneuver_cmds[0]['commands'] = ''.join(splitx) 
            
        else: # It's a bogus maneuver entry - the user didn't specify good Q's
            self.logger.warning("Bogus Maneuver Entry! Quaternions badly specified: \n"
                                "Bad Q's: %g, %g, %g, %g " % (q1, q2, q3, q4) +
                                "...therefore bogus pitch and roll: %g, %g" % (pitch, roll))

        # Return the list of processed maneuver commands
        return processed_maneuver_cmds


#-------------------------------------------------------------------------------
#
# Process_LTCTI - process the submitted LTCTI line from the NLET file
#
#-------------------------------------------------------------------------------
    def Process_LTCTI(self, ltcti_event, trim_date):
        """
            Inputs: ltcti_event - Event line from the NLET file, split on spaces, indicating
                                  a LTCTI entry

                                - format: 2020:147:02:08:00    LTCTI   1527     1_4_CTI    000:16:00:00
                                          event start date      type   CAP #    CLD file   Length of LTCTI

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
        # Split the NLET event line on spaces
        ltcti_event_split = ltcti_event.split()

        RTS_start_date = ltcti_event_split[0]
        CAP_num  = ltcti_event_split[2]
        RTS_name = ltcti_event_split[3]
        NUM_HOURS = ltcti_event_split[4]

        # SCS NUm is the SCS slot specified in the FOR request. If you are not processing
        # FOT requests then set it to 135
        SCS_NUM = 135

        self.logger.info("LTCTI Measurement Detected: %s \n" % (ltcti_event))

        # Process the specified RTS file and get a time-stamped numpy array of the commands
        ltcti_cmd_list = LTCTI_RTS.processRTS(RTS_name, SCS_NUM, NUM_HOURS, RTS_start_date)

        # Make a list of the needed commands from the mnemonic column of ltcti_cmd_list
        raw_ltcti_cmd_list = [eachcmd for eachcmd in ltcti_cmd_list['mnemonic'] ]

        # Create the raw command array
        LTCTI_cr_cmds = self.Process_Cmds(raw_ltcti_cmd_list, RTS_start_date)

        # Now Process_Cmds has a rudimentary numbering scheme - it increments the time
        # by a dt and each successive command is dt seconds later. 
        #
        # For Processing LTCTI this isn't good enough. However the array
        # Returned by RTS.processRTS does have the correct timing so we execute
        # a simple substitution
        for eachRTScmd, eachCRcmd in zip(ltcti_cmd_list, LTCTI_cr_cmds):
            # Modify the date and time columns
            eachCRcmd['date'] = Time(eachRTScmd['date'], format = 'yday', scale = 'utc').yday
            eachCRcmd['time'] = eachRTScmd['time']
            # Now modify the CR command string
            eachCRcmd['commands'] = eachCRcmd['date'].ljust(22) + eachCRcmd['commands'][22:] 

        self.logger.debug('Nominal LTCTI start and stop dates: %s   %s' % (LTCTI_cr_cmds[0]['date'], LTCTI_cr_cmds[-1]['date']) )

        self.logger.debug(LTCTI_cr_cmds)


        # We need to find the first ACISPKT command in the master list that
        # comes after the start of the LTCTI first command, and is ALSO
        # a Stop Science ('AA00000000').
        # IMPORTANT: The LTCTI run may have started in the Continuity load
        #            but it will end either because it runs to completion with
        #            it's own Stop Science commands, OR a Stop Science in the review load
        # Obtain the start and end times of the timed LTCTI command set
        
        ltcti_cmd_list_start_time = LTCTI_cr_cmds[0]['time']
        ltcti_cmd_list_stop_time = LTCTI_cr_cmds[-1]['time']

        # Next, make a list all ACISPKTcommands in the master list which
        # are between the LTCTI start time and the LTCTI stop time, inclusive
 
        # Make a copy of the master list
        x = self.master_list 

        # Trim all commands prior to ltcti_cmd_list_start_time
        x = self.Trim_bs_cmds_Before_Time(ltcti_cmd_list_start_time, x)   

        # Trim all commands AFTER ltcti_cmd_list_stop_time
        x = self.Trim_bs_cmds_After_Time(ltcti_cmd_list_stop_time, x) 

        self.logger.debug('Number of commands left in x after trim: %d' % ( len(x) ) )

        # x now contains any load commands that occured during the LTCTI run
        # Find out if there are any Stop Sciences in the resultant trimmed list

        # If there are no commands left after the trim then you can use the entire
        # LTCTI command sequence you've built.
        if x.size == 0:
            self.logger.info("LTCTI Measurement runs to completion\n" )
        else:
            # There are load commands within the start and stop time of the LTCTI.
            # They may or may not terminate the LTCTI before it's nominal run.
            # Find those instances (if any) where a load command, within the trimmed 
            # set, is a Stop Science (AA0000000).  If you find one then chop any 
            # LTCTI commands which come on or after that Stop Science.
            # 
            # Collect any stop science commands within the trimmed 
            aa_instances = [cmd_index for cmd_index, eachcmd in enumerate(x) if 'AA00000000' in eachcmd['commands']]

            # If there are no Stop Science commands within the start and stop of the
            # LTCTI run then th4e LTCTI runs to completion
            if len(aa_instances) == 0:
                self.logger.info("Load commands exist within the LTCTI but LTCTI Measurement runs to completion\n" )
            else:
                self.logger.debug('LTCTI Interrupted at: %s with load command:\n    %s' % (x[aa_instances[0]]['date'], x[aa_instances[0]]['commands']))
                # The LTCTI was terminated before completion. Remove all LTCTI commands
                # on or before the Stop Science.  Get th eindex of the terminating Stop Science.
                cut_index = aa_instances[0]

                # record the cut time
                ltcti_cut_time = x[cut_index]['time']
                ltcti_cut_date = x[cut_index]['date']

                # Now trim the sequence of ltcti commands
                LTCTI_cr_cmds = self.Trim_bs_cmds_After_Time(ltcti_cut_time, LTCTI_cr_cmds)

                # Tell the user that the LTCTI was cut short
                self.logger.info("LTCTI Measurement Cut Short at %s %s\n" % (ltcti_cut_date, ltcti_cut_time))

        # They are out of time order but will be sorted in the calling routine.
        return LTCTI_cr_cmds


    #-------------------------------------------------------------------------------eachcmd['date']
    #
    # Process_Power_Cmd
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
        # split the command on spaces
        power_cmd_list = power_cmd_event.split()
        
        # Log that you see it
        self.logger.info("Power Command Detected %s power cmd: %s" % (power_cmd_list[0], power_cmd_list[1]) )

        # Create the time stamped power command set using date in the NLET line as the start time
        new_power_cmd = self.Process_Cmds([power_cmd_list[1]], power_cmd_list[0])

        return new_power_cmd

    #-------------------------------------------------------------------------------
    #
    # Process_Cmds
    #
    #-------------------------------------------------------------------------------
    def Process_Cmds(self, cmd_list, BEGIN_date):
        """
        This method will take a list of command names and a starting time stamp and generate
        an array of commands with the date string of the command, the date and the time 
        columns all updated
             Input: cmd_list - list of strings showing the command sequence
                                e.g. [ 'SIMTRANS', 'AA00000000', 'AA00000000', 'WSPOW0002A']

                    BEGIN_date - SCS-107/STOP or sequence start date(s) in Chandra DOY format

            Output: processed_cmd_array
                     - array of commands adjusted to the given BEGIN date
        """

        # Make a list of the actual command strings, in raw form, 
        # for each command in the cmd_list using the CR_cmds dictionary
        raw_cmds = [self.CR_cmds[eachcmd] for eachcmd in cmd_list]

        # Calculate the times and the dates: split the command string on spaces, calculate the time and add the date
        cmd_times = [round(Time(eachdate.split()[0], format = 'yday', scale = 'utc').cxcsec,1) for eachdate in raw_cmds]
        cmd_dates = [eachdate.split()[0] for eachdate in raw_cmds]

        # Now make a numpy array with two columns: cmd strings and times
        cmd_array = np.array(list(zip(raw_cmds, cmd_times, cmd_dates)) , dtype = self.CR_DTYPE) 

        # Now substitute the dates and times in the command portion and the
        # time column of the array

        # The starting time for the first scs107 command will be at the stop time
        # obtained from the ACIS-Continuity.dat file. Run Time on the date string
        # to arrange for 3 places to the right of seconds.
        base_date = Time(BEGIN_date, format = 'yday', scale = 'utc').yday
        base_time = Time(BEGIN_date, format = 'yday', scale = 'utc').cxcsec

        #  Each subsequent time will be 1 second after the last
        dt = 1.0

        # Populate the date and time slots of each command incrementing the times
        # by one second
        for eachcmd in cmd_array:
            eachcmd['date'] = base_date
            eachcmd['time'] = base_time

            # Substitute the updated date into the command string
            eachcmd['commands'] = base_date.ljust(22) + eachcmd['commands'][22:] 

            # Increment the base time by one second
            base_time += dt

            # Convert the new time to yday format
            base_date = Time(base_time, format = 'cxcsec', scale = 'utc').yday

        # Return the time stamped scs-107 command set
        return cmd_array


    #--------------------------------------------------------------------------
    #
    # WriteCommands to file - Write the specified commands to a file
    #
    #--------------------------------------------------------------------------
    def Write_Commands_to_File(self, cmd_list, outfile_path, outfile_mode, comment = None):
        """
        This method will write the specified command list out to a file whose path is specified
        in outfile_path.  

        The commands are either written to a new file or appended to an existing file as
        indicated by outfile_mode

        Whether or not this is an original command list or a combined one it immaterial.

            INPUTS: command list - table fo commands
                    outfile_path - full output file specification
                    outfile_mode - "w" or "a"
                    comment      - Comment to be written out to the file

           OUTPUTS: Nothing returned; file written.
        """
        # Open up the file for writing
        combofile = open(outfile_path, outfile_mode)

        # Now output pertinent info from the command list. Make it look like the
        # original CR backstop command file but add the extra info such as time 
        # at the end after the double bar
        for eachcmd in cmd_list:
            combofile.write(eachcmd['commands']+ '\n')

        # Output the comment if one was given
        if comment is not None:
            combofile.write('\nComment: '+comment+'\n')
        # Done with the file; close it.
        combofile.close()


    #-------------------------------------------------------------------------------
    #
    # Trim_bs_commands_After_Date - Given a list of backstop commands, remove any command
    #                               prior to the specified time
    #
    #                 INPUTS: Chandra time 
    #                         List of backstop commands.
    #
    #                OUTPUTS: Trimmed list of backstop commands
    #-------------------------------------------------------------------------------
    def Trim_bs_cmds_After_Time(self, trim_time, cmd_list):
        """
        Given an astropy table of backstop commands, remove any command on
        or *AFTER* the specified time

        INPUTS: Trim time in Chandra time (Chandra Seconds)

                List of backstop commands.

        OUTPUTS: The trimmed input table

        NOTE: The input table is modified 
        """
        # Can only trim if the specified trim time is within the cmd_list
        # first time and last time
        if (trim_time >= cmd_list[0]['time']) and \
           (trim_time <= cmd_list[-1]['time']):
            # Get the index of the first row whose time value is greater than trim time
            cut_index = np.where(cmd_list['time'] > trim_time)[0][0] 

            # Remove all rows starting from that index to the end of the list
            cmd_list = np.delete(cmd_list, slice(cut_index, len(cmd_list)), axis = 0)
        # ....otherwise you will return the untouched command list

        # Return the possibly trimmed command list
        return cmd_list



    #-------------------------------------------------------------------------------
    #
    # Trim_bs_cmds_Before_Date - Given a list of backstop commands, 
    #                            remove any command on or 
    #                            prior to the specified time
    #
    #                 INPUTS: Chandra  time in seconds
    #                         List of backstop commands.
    #
    #                OUTPUTS: Trimmed list of backstop commands
    #-------------------------------------------------------------------------------
    def Trim_bs_cmds_Before_Time(self, trim_time, cmd_list):
        """
        Given an astropy table of backstop commands, remove any command 
        *PRIOR* to or on the specified time

        INPUTS: trim time in Chandra time (Chandra Seconds)

                List of backstop commands.

        OUTPUTS: The trimmed input table

        NOTE: The input table is modified, so unless you want the master
              list trimmed, don't send that one in.
        """
        # Can only trim if the specified trim time is within the cmd_list
        # first time and last time
        if (trim_time >= cmd_list[0]['time']) and \
           (trim_time <= cmd_list[-1]['time']):

            # Get the index of the first row whose time value is greater than trim time
            cut_index = np.where(cmd_list['time'] < trim_time )[0][-1]

            # Remove all rows starting from index 0 to the specified index 
            cmd_list = np.delete(cmd_list, slice(0, cut_index+1), axis = 0)

        # ....otherwise you will return the untouched command list

        # Return the trimmed list
        return cmd_list


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
    def get_BackChain_List(self, base_load_dir, chain_length):
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
        base_load_week = os.path.split(base_load_dir)[0].split('/')[-1]

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
                                             continuity_info[1],
                                             continuity_info[2],
                                             continuity_load_path) ],
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
    #                             file for any NLET event that occurred between the 
    #                             start and stop times.
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

        tstart and tstop are expected to be in Chandra seconds
        """
        # Initialize and empty Event List
        event_list = []
        # Convert the input tstart and tstop to seconds - this allows the
        # user to input either seconds or DOY format - whichever is more
        # convenient.

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
                   (Time(splitline[0], format='yday', scale='utc').cxcsec > tstart) and \
                   (Time(splitline[0], format='yday', scale='utc').cxcsec < tstop):

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
                                                           Time(packet_time, format = 'yday', scale = 'utc').cxcsec,
                                                           cmd_type,
                                                           cmd) ],
                                                       dtype = self.ACIS_specific_dtype) ]

        # Finished reading and processing the file
        bsdf.close()

        # Return the backstop command array
        return ACIS_specific_cmds

    ############################################################################
    #
    #                    Attribute Sets and Gets
    #
    ############################################################################

    #---------------------------------------------------------------------------
    #
    # get_review_tstart
    #
    #---------------------------------------------------------------------------
    def get_review_tstart(self,):
        """
        Return the contents of attribute: review_file_tstart
        """
        return self.review_file_tstart

    #---------------------------------------------------------------------------
    #
    # get_review_tstop
    #
    #---------------------------------------------------------------------------
    def get_review_tstop(self,):
        """
        Return the contents of attribute: review_file_tstop
        """
        return self.review_file_tstop
    
    #--------------------------------------------------------------------------------
    #
    # Extract_by_Token - New BSH method
    #
    #--------------------------------------------------------------------------------
    def Extract_by_Token(self, token_list, cmd_array):
        """
        Given a list of tokens, and an array of backstop commands, of the same dtype
        as Backstop History master_list, extract any row whose "commands" column
        contains any of the tokens in the token list.  Place the extracted rows in an
        array of the same dtype and return that array
    
        inputs: token_list - list of tokens to search for [e.g. ["OORMPDS", "EPERIGEE"]
                                       The tokens must be strings.
    
                   cmd_array - An array which is a history of CR*.backstop commands 
                                       which was assembled by Backstop History and is of the
                                       same dtype as Backstop History master_list
    
        outputs: - extracted_array - Array containing any extracted rows. The array
                                                      be empty
        """
        extracted_array = np.array([], dtype = cmd_array.dtype)
        
        for each_cmd in cmd_array:
            if [True for token in token_list if token in each_cmd["commands"] ]:
                new_cmd = np.array( [ ( each_cmd["commands"],
                                                         each_cmd["time"],
                                                         each_cmd["date"] ) ], dtype = cmd_array.dtype)
                
                extracted_array = np.append(extracted_array, new_cmd, axis=0) 
        
        # Return the extracted array
        return extracted_array
    
