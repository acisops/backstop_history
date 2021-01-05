import numpy as np
import os
import re

# Import the class that contains individual command data for all
# the commands which are relevant to LTCTI runs
from backstop_history import LTCTI_ACIS_commands

from cxotime import CxoTime


class LTCTI_RTS(object):
    """
    This class allows the user to convert a FOT, ACIS, LTCTI, RTS file, along with
    information found in a FOT Request, and generate a list of dicts where each line in the RTS file is
    processed and converted to a dict in the SKA.parse format. Each dict is appended
    to an output list.

    The format of the RTS file and the FOT request are described in OP-19.

    An example line from the LTCTI FOT request looks like this:

        RTSLOAD,1_CTI06,SCS_NUM=135,NUM_HOURS=001:15:00:00

    This line tells you that:

      1) You should use the 1_CTI06.RTS RTS file (this is a 6 chip LTCTI run)
      2) The commands will be placed in SCS slot 135
      3) The length of the CTI run - if undisturbed by a Return to Science - is
         1 day, 15 hours 0 minutes and 0 seconds.

    A Sample input line from the .RTS file looks like this:

         ACIS,WSPOW0CF3F,DELTA=00:00:01.000

    This line informs you that it's an ACIS command, that the command mnemonic is 
    WSPOW0CF3F, and (VERY important) that the command should be executed 1 second after the 
    previous command (or 1 second after the start time of the RTS execution for the first
    command in the RTS).

    The method processRTS() takes the start time and any DELTA parameter in the RTS line
    and calculates the execution time of the command. If no DELTA appears in the line
    being processed then the last computed time is used.


    You can process an actual FOT request, like the example above, or get the values from
    some other means, such as the Non-Load Event Tracking file.

    The conversion is a three step process:

    First, the FOT request file is read and information extracted using

        parse_FOT_request(fot_request).

    Among other things, this file tells you which CTI RTS  file should be used. You can 
    Yuo can skip this step and proceed directly to Step 2 if you have obtained the necessary
    information in some other way.

    Second, the processRTS() method is used to open the file, read
    and process each line and write the results into a numpy array.  The array DTYPE is:

dtype=[('date', 'S20'), ('time', '<f8'), ('statement', 'S20'), ('mnemonic', 'S20'), ('substitution_parameter', 'S20'), ('substitution_parameter_value', 'S20'), ('DELTA', '<f8'), ('SCS_NUM', 'S5')])
    
    Every command is converted into an array entry.  The column names used in the DTYPE 
    were obtained from OP-19.

    The user can stop there or proceed to the third step which is to convert each 
    ACIS entry in the array into a dict in the SKA.Parse format using 

    convert_ACIS_RTS_to_ska_parse(). 

    Important Note: This class is written with a hook  called:

                            convert_RTS_to_ska_parse(self, RTS_cmds)

                    to translate all of the command lines into dicts, but it's only a stub.
                    convert_ACIS_RTS_to_ska_parse() translates only the ACIS commands
                    in the RTS to SKA.Parse dicts and that's fully functional. 
                    If you wish to translate them all, then use this class as a Base class
                    and overload the convert_RTS_to_ska_parse() method.


    inputs:  1) RTS file location
                  - Directory in which your RTS files canbe found.
                   At present there are only three:
                   
                   1_4_CTI.RTS
                   1_5_CTI.RTS
                   1_CTI06.RTS
                   
              3) Start Time for the execution of the RTS in Chandra DOY format

    outputs:  1) numpy array where each line is a processed line of the RTS file
              2) List of dicts for each ACIS command where the dicts are in SKA.Parse
                 format.

    """
    # Constructor
    def __init__(self, RTS_file_loc):
        # Create the rec array dtype for the mnemonics.hrd file read
        self.RTS_dtype = [('date', '|U20'),
                          ('time','<f8'),
                          ('statement', '|U20'),           
                          ('mnemonic', '|U20'), 
                          ('substitution_parameter',  '|U20'),
                          ('substitution_parameter_value',  '|U20'),
                          ('DELTA','<f8'),
                          ('SCS_NUM', '|U5')]
        self.load_type = None
        self.RTS_name = None
        self.SCS_NUM = '135'
        self.NUM_HOURS = None
        self.CAP_num = None

        # Directory where the class can find the CTI RTS files and FOT request file
        self.RTS_file_loc = RTS_file_loc

        self.present_time = None      # Variable used to store the time in seconds
                                      # of the last processed RTS command

        # Create an instance of the ACIS LTCTI CLD file commands class
        self.ACIS_cmds = LTCTI_ACIS_commands.LTCTI_ACIS_commands()

    #-------------------------------------------------------------------------------
    #   
    #    Method parse_FOT_REQUEST
    #
    #-------------------------------------------------------------------------------
    def parse_FOT_request(self, fot_request_file):
        """
        Takes the request from the FOT to run one of the 3 LTCTI RTS's and
        parses it out into the constituent values - many of which are needed
        by this class to expand the general RTS into a specific one.

        The request is in the form of a 3 line file

            input: fot_request_file - full path to the Fot Request file to be 
                   processed. 
                       - The FOT request file is of the form:

                          SCS_CATEGORY, OBSERVING

                          RTSLOAD,1_CTI06,SCS_NUM=135,NUM_HOURS=001:15:00:00

            output - this method fills the attributes as defined in the constructor
                     specifically:

                     load_type
                     RTS_name
                     SCS_NUM
                     NUM_HOURS
        """
        # Define the regex matches
        rtsload_match = re.compile('RTSLOAD')
        name_match = re.compile('1_CTI0')
        scs_match = re.compile('SCS_NUM=')
        num_hours_match = re.compile('NUM_HOURS=')

        # Open the request file for reading
        fot_req = open(self.RTS_file_loc+fot_request_file, 'r')

        # Read eachline and toss until you get to the line that begins with "RTSLOAD"
        for eachline in fot_req:
            # Split on commas
            split_line = eachline.split(',')
            # If RTSLOAD appears in the line
            if any(filter(rtsload_match.match, split_line)):
                self.load_type = 'RTSLOAD'
                self.RTS_name = list(filter(name_match.match, split_line))[0]
                self.SCS_NUM = list(filter(scs_match.match, split_line))[0].split('=')[-1]
                self.NUM_HOURS = list(filter(num_hours_match.match, split_line))[0].split('=')[-1]
        # Done with the FOT request file - close it
        fot_req.close()

    #-------------------------------------------------------------------------------
    #   
    #    Method convert_RTS_DELTA_to_secs
    #
    #-------------------------------------------------------------------------------
    def convert_RTS_DELTA_to_secs(self, time_string):
        """
        This method takes a string in the format:

              ddd:hh:mm:ss

        and converts it into seconds.

              ddd = number of days (NOT DOY!!!!)
               hh = hours
               mm = minutes
               ss = seconds

        IMPORTANT NOTE:  These input strings are NOT DOY time strings.  I.E. "ddd"
                         is NOT DAY of year.  "ddd" is equal to a number of days.
    
        Examples: 001:00:00:00 input will give you 86400.0 seconds
                  000:02:00:00 input will give you  7200.0 seconds.
                  000:00:00:01 input will give you     1.0 seconds
    
        Now you can send the NUM_HOURS value in the FOT request as is - it is of 
        the format ddd:hh:mm:ss.
    
        But you'd also like to convert the  DELTA= values that you see in the RTS file.
        You can - just be sure you concatenate '000:' to the beginning of the DELTA value
        first.  DELTA= format is hh:mm:ss.sss
    
        """
        # Split the string on colons; multiply each value in the resulting list by
        # its unit conversion in seconds.
        duration_secs = sum(float(n) * m for n, m in zip(time_string.split(':'), (86400.0, 3600.0, 60.0, 1)))
    
        # return the converted duration
        return duration_secs
    
    #-------------------------------------------------------------------------------
    #   
    #    Method process_RTS
    #
    #-------------------------------------------------------------------------------
    def processRTS(self, RTS_load, SCS_NUM, NUM_HOURS, RTS_start_date):
        """
        This method opens the specified RTS, reads each line and creates an
        array which contains the values in the line plus time stamps the line.
        

        inputs:      RTS_load : Name of the LTCTI RTS file (e.g. 1_4_CTI)
                     SCS_NUM  : The SCS number this RTS file was run in (e.g. 135)
                    NUM_HOURS : a string in the FOT request format:  ddd:hh:mm:ss
               RTS_start_date : Start of the LTCTI in DOY format

        ACIS LTCTI RTS files contain comma separated lines, and each line entry
         can have 2,3 or 4 columns.
        
        Samples are:  /CMD, OORMPEN
                      ACIS,WSVIDALLDN, DELTA=00:00:01.000
                      /CMD, 2S2STHV, 2S2STHV2=0, DELTA=00:00:01.000
        
          Any line that does not start with 'ACIS' is logged in the array but not used by ACIS
          Ops other than to extract and use a DELTA if any.
              
        The important point about using the DELTA's is that if one exists, you MUST apply
        The delta time to the ongoing time stamp before you save the time for that line. 
        
        For example, suppose the time stamp is 2018:001:00:00:00.00 and you are 
        processing this line:
        
            ACIS,WSPOW0CF3F,DELTA=00:00:01.000
        
        The time stamp for that command is the present time stamp plus the DELTA in that line or:
        
            2018:001:00:00:01.00
        
        another way of saying it is that the DELTA is the delay between the previous command and
        the one you are processing.
        
        If there is no DELTA use the present time stamp value
        
        """
        # Compile the regex match criteria for the lines in the CTI RTS files
        delta_match = re.compile('DELTA=')
        equal_match = re.compile('=')
        cmd_statement_match = re.compile('/CMD')
        acis_statement_match = re.compile('ACIS')
        
        
        # Convert the RTS start date into seconds. We will use this
        # to calculate the time of each command in the RTS
        present_date = RTS_start_date
        present_time = CxoTime(RTS_start_date).secs
        
        # Calculate the duration of the CTI run, in seconds, as if it was NOT interrupted
        # by a Return To Science....i.e. it ran to completion and was
        # followed by the Perigee passage
        cti_duration_secs = self.convert_RTS_DELTA_to_secs(NUM_HOURS)
        
        # Convert the RTS start date to seconds.
        present_time = CxoTime(RTS_start_date).secs
                
        # Create an empty array with the RTS_dtype
        RTS_cmds = np.array([], dtype=self.RTS_dtype)
              
        # Form the full path to the appropriate RTS file
        rts_file_path = os.path.join(self.RTS_file_loc, self.RTS_name+'.RTS')
        # Open the specified RTS file for this Long Term CTI run
        rts_load = open(rts_file_path, 'r')
        
        # Process each command in the RTS
        # You want to fill out the RTS_dtype to the degree that you can
        #
        # So for each line in the RTS file.....
        for eachline in rts_load:
            # Ignore the line if it is a comment or a blank line
            if (eachline[0] != '!') and (eachline[0:2] != '\n'):

                # Split and join the lines while eliminating all whitespace that
                # occasionally occurs with the list items
                split_line = ''.join(eachline.split())

                # Now split the line on commas
                split_line = split_line.split(',')
                
                # /CMD or ACIS STATEMENT
                # Find out what the statement value is (/CMD or ACIS)
                if any(filter(cmd_statement_match.match, split_line)):
                    # It's a /CMD line so set the statement to /CMD
                    statement = '/CMD'
                elif any(filter(acis_statement_match.match, split_line)):
                    # It's an ACIS line so set the statement to ACIS
                    statement = 'ACIS'
                else:
                    # It's neither a /CMD or ACIS line so set the statement to None
                    statement = None

                # MNEMONIC
                if statement is not None:
                    # Find the position in the list of the statment
                    # It's usually the first line but don't make assumptions
                    statement_pos = split_line.index(statement)
                    # The Mnemonic comes immediately after the statement
                    mnemonic = split_line[statement_pos+1]

                # Grab all list items that have an equal sign
                equal_items_list = list(filter(equal_match.findall, split_line))
        
                # Is there a DELTA= in any list item?  
                d_match = list(filter(delta_match.findall, equal_items_list))
        
                # If so, extract the time string in the DELTA entry, convert
                # it to seconds, and add those number of seconts to the present_time
                if d_match:
                    # You have a match convert and save the value..........
                    delta_string = d_match[0].split('=')
                    # Check to see if the Time String is &NUM_HOURS&, If it is,
                    # then use cti_duration_secs
                    if delta_string[-1] == '&NUM_HOURS&':
                        dt = cti_duration_secs
                    else:
                        # Otherwise use the value represented by the string
                        dt = self.convert_RTS_DELTA_to_secs('000:' + delta_string[-1])
        
                    # In either case, advance the prewsent time by dt
                    present_time += dt
                    # Get the index of the DELTA item in the equal_items_list
                    # and remove that item from the list
                    equal_items_list.remove(d_match[0])
                else:
                    dt = 0.0
        
                # Whether you had a delta or not, convert the present time to a date
                present_date = CxoTime(present_time).date 
        
                # Now if equal_items_list is not empty, after you removed the DELTA
                # line, then the RTS line had a 
                # substitution parameter and a supbstitution parameter value. The
                # format of the line is "substitution_parameter=value'.
                if equal_items_list:
                    # There is a substitution value in there so set the
                    # variables appropriately.
                    sub_split = equal_items_list[0].split('=')
                    substitution_parameter = sub_split[0]
                    substitution_parametr_value = sub_split[1]
                else:
                    # The list was empty so there was no substitution parameter
                    # so therefore set the two variables to None
                    substitution_parameter = None
                    substitution_parametr_value = None
        
                # Now you have all the information necessary to put an entry into the array
                RTS_cmds = np.r_[RTS_cmds,
                                  np.array( [ (present_date,
                                               present_time,
                                               statement,
                                               mnemonic,
                                               substitution_parameter,
                                               substitution_parametr_value,
                                               dt,
                                               SCS_NUM) ],
                                            dtype = self.RTS_dtype)  ]
        
        # Done with the RTS file - close it
        rts_load.close()
        
        # Return the RTS_cmds array
        return RTS_cmds
    
    
    
    
    #-------------------------------------------------------------------------------
    #   
    #    Method Convert_ACIS_RTS_to_ska_parse
    #
    #-------------------------------------------------------------------------------
    def convert_ACIS_RTS_to_ska_parse(self, RTS_cmds):
        """
        This method will convert the input array which is a sequence of 
         processed RTS commands (as output by processRTS) into a list of commands
         in the Ska.Parse/bs_cmds format (i.e. a list of dicts). 
  
         The timing of the commands was already taken care of by processRTS.

        INPUT: 1) Array of RTS file format commands as returned by processRTS

                  Input commands look like this:

array([ ('2018:001:00:00:00.00', 631152069.184, '/CMD', 'OORMPEN', 'None', 'None', 0.0, '135'),
       ('2018:001:00:00:01.00', 631152070.184, 'ACIS', 'WSVIDALLDN', 'None', 'None', 1.0, '135'),
       ('2018:001:00:00:02.00', 631152071.184, '/CMD', '2S2STHV', '2S2STHV2', '0', 1.0, '135'),
....................
     dtype=[('date', 'S20'), ('time', '<f8'), ('statement', 'S20'), ('mnemonic', 'S20'), ('substitution_parameter', 'S20'), ('substitution_parameter_value', 'S20'), ('DELTA', '<f8'), ('SCS_NUM', 'S5')])

               
        OUTPUT: List of dicts where each entry is an ACIS command converted to 
                Backstop File format

        IMPORTANT NOTE:  Any non-ACIS command ( i.e. statement = '/CMD') is ignored.
                         so if you need to convert all the commands in the RTS,
                         then use this class as a base class and overload the 
                         convert_RTS_to_ska_parse() method
        """    
 
        # Create an instance of LTCTI_RTS_backstop_commands
        command_data = LTCTI_ACIS_commands.LTCTI_ACIS_commands()
               
        # Empty output list
        RTS_ska_bs = []
    
        # Initialize the step to 1. It can be used as a delta by
        # subsequent processing of the output
        step = 1
        
        # Process each command in the input array RTS_cmds
        for eachcommand in RTS_cmds:
            if eachcommand['statement'] == 'ACIS':
        
                tlmsid = eachcommand['mnemonic']
                CMDS = command_data.cmd_vals[eachcommand['mnemonic']]['CMDS']
                WORDS = command_data.cmd_vals[eachcommand['mnemonic']]['WORDS']
                packet = command_data.cmd_vals[eachcommand['mnemonic']]['PACKET']
            
                # Form a command with the parameters obtained from the RTS file
                new_cmd = {'idx': -1,
                           'date': eachcommand['date'],
                           'type': 'ACISPKT',
                           'tlmsid': tlmsid,
                           'scs': self.SCS_NUM,
                           'step': step,
                           'time': -1,
                           'timeline_id': 0,
                           'vcdu': 0000000,
                           'params': {'cmds': CMDS,
                                      'words': WORDS,
                                      'packet(40)': packet}}

                # Now convert the date to seconds and modify the dictionary
                new_cmd['time'] = eachcommand['time']
        
                # Append the new command to the output list
                RTS_ska_bs.append(new_cmd)

            # Always increment the step no matter what the command
            # For the purposes of Thermal Models, the step value is
            # irrelevant as is the VCDU counter.
            step +=1

        # Return the updated dictionary
        return RTS_ska_bs
    
    
    #-------------------------------------------------------------------------------
    #   
    #    Method Convert_RTS_to_ska_parse
    #
    #-------------------------------------------------------------------------------
    def convert_RTS_to_ska_parse(self, RTS_cmds):
        """
        This is the method you overload if you want to convert every command
        in the RTS_cmds array into SKA.Parse format
        """
        pass
