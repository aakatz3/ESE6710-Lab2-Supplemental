# Flags
PLOT = True
SHOW_PLOT = False
CLEAR_PREVIOUS = True
SHORT = True
LONG = False

import pyvisa as visa
import time
import datetime as dtime
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib as p
import pandas as pd
import shutil
import tkinter.messagebox as mb
import sys
import eseries


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


dir = p.Path('data')
wavepath = p.Path(dir, 'waveform')
sspath = p.Path(dir, 'scope')
if (PLOT):
    pltpath = p.Path(dir, 'plt')

if CLEAR_PREVIOUS and os.path.isdir(dir.resolve()):
    shutil.rmtree(dir.resolve())

os.makedirs(dir, exist_ok=True)
os.makedirs(wavepath, exist_ok=True)
os.makedirs(sspath, exist_ok=True)
if (PLOT):
    os.makedirs(pltpath, exist_ok=True)

# start of Experiment

R_LOADS = [5] #, 2.5, 10, 20, 50, 100, 200, 500]
for x in eseries.erange(eseries.E6,1,10):
    R_LOADS.append(x)
FREQ_RANGE = (100e3, 500e3)
MANUAL_FREQS = [129.7e3, 139.7e3]
FREQ_STEP = 1e3

freqs = np.append(np.arange(FREQ_RANGE[0], FREQ_RANGE[1], FREQ_STEP),
                  FREQ_RANGE[1])
freqs = np.append(freqs, MANUAL_FREQS)
freqs.sort()

# Limit for quick testing
# freqs = freqs[0:2]
# R_LOADS = R_LOADS[0:3]

offset = 1
v_out = []
p_out = []
i_out = []

try:
    rm = visa.ResourceManager()
    v33521A = rm.open_resource('USB0::2391::5639::MY50000944::0::INSTR')
    EL34143A = rm.open_resource('USB0::10893::14338::MY61001509::0::INSTR')
    MSO7034B = rm.open_resource('USB0::2391::5949::MY50340240::0::INSTR')
    EDUX1052G = rm.open_resource('USB0::0x2A8D::0x039B::CN60201142::0::INSTR')
    EDU34450A = rm.open_resource('USB0::0x2A8D::0x8E01::CN62180094::0::INSTR')

    MSO7034B.timeout = 10000
    EDUX1052G.timeout = 10000
    EDU34450A.timeout = 15000

    with p.Path(dir, 'instruments.log').open('w', encoding='cp1252') as log:
        log.write('Timestamp: ' +
                  dtime.datetime.now().astimezone().isoformat() + os.linesep)
        print('Instruments Utilized:')
        for inst in [v33521A, EL34143A, MSO7034B, EDUX1052G, EDU34450A]:
            try:
                inst.write('*CLS')
                idn =              inst.query('*IDN?').strip('\r').strip('\n')
                log.write(idn + os.linesep)
                print(' - ' + idn)
                if inst != MSO7034B:
                    inst.write('*RST')
            except BaseException as e:
                print(e)
            finally:
                if inst == MSO7034B:
                    inst.write('*CLS')

    time.sleep(2)
    # Check that input voltage is there
    EDU34450A.write(':FORMat:OUTPut %d' % (1))
    vol = EDU34450A.query_ascii_values(':MEASure:PRIMary:VOLTage:DC?')[0]
    if round(vol) != 24:
        err = 'Invalid Setup:\n'+\
             'Input voltage should be 24\n'+\
             f'but is measured as {vol}'
        eprint(err.replace('\n', ' '))
        mb.showerror(title='Setup Error', message=err)
        exit(-1)
    else:
        print(f'Input voltage is {vol}')
    # Setup Electronic Load
    v33521A.write(':OUTPut1:STATe %d' % (0))
    EL34143A.write(':SOURce:FUNCtion %s' % ('RESistance'))
    EL34143A.write(':SOURce:RESistance:LEVel:IMMediate:AMPLitude %G' % (1000))
    EL34143A.write(':OUTPut:STATe %d' % (1))
    time.sleep(2)
    EL34143A.write(':SOURce:RESistance:LEVel:IMMediate:AMPLitude %G' %
                   (R_LOADS[0]))
    EL34143A.write(':OUTPut:STATe %d' % (1))
    # Setup DMM
    EDU34450A.write(':FORMat:OUTPut %d' % (1))
    EDU34450A.write(':TRIGger:SOURce %s' % ('IMMediate'))
    EDU34450A.query_ascii_values(':MEASure:PRIMary:CURRent:DC?')

    EDU34450A.query_ascii_values(':MEASure:SECondary:VOLTage:DC?')
    # Setup wavegen
    v33521A.write(':OUTPut:LOAD %s' % ('INFinity'))
    v33521A.write(':SOURce:FUNCtion:SHAPe:SQUare:DCYCle %G' % (1.0))
    v33521A.write(':OUTPut1:STATe %d' % (1))
    v33521A.write(':SOURce:APPLy:SQUare %G,%G,%G' % (FREQ_RANGE[0], 5.0, 2.5))

    # Slowly ramp duty cycle to 50% to prevent overloading E-load
    for x in range(1, 51):
        v33521A.write(':SOURce:FUNCtion:SHAPe:SQUare:DCYCle %G' % (x))
        time.sleep(0.2)

    labels = ['V_SW', 'I_IN', 'I_OUT', 'I_P', 'V_SEC', 'V_OUT']
    # Setup scopes
    try:
        MSO7034B.write(':HARDcopy:INKSaver %d' % (0))
        MSO7034B.write(':CHANnel1:LABel "%s"' % (labels[0]))
        MSO7034B.write(':CHANnel2:LABel "%s"' % (labels[1]))
        MSO7034B.write(':CHANnel3:LABel "%s"' % (labels[2]))
        MSO7034B.write(':CHANnel4:LABel "%s"' % (labels[3]))
        MSO7034B.write(':DISPlay:LABel %d' % (1))
        MSO7034B.write(':TRIGger:EDGE:SOURce %s' % ('EXTernal'))
        MSO7034B.write(':TRIGger:EDGE:LEVel %G' % (0.5))
        MSO7034B.write(':TRIGger:EDGE:SLOPe %s' % ('POSitive'))
        MSO7034B.write(':CHANnel1:DISPlay %d' % (1))
        MSO7034B.write(':CHANnel2:DISPlay %d' % (1))
        MSO7034B.write(':CHANnel3:DISPlay %d' % (1))
        MSO7034B.write(':CHANnel4:DISPlay %d' % (1))
        MSO7034B.write(':TIMebase:MAIN:SCALe %G US' % (2.0))
        MSO7034B.write(':CHANnel1:SCALe %G' % (25.0))
        MSO7034B.write(':CHANnel1:OFFSet %G' % (80))
        MSO7034B.write(':CHANnel2:SCALe %G MV' % (200.0))
        MSO7034B.write(':CHANnel2:OFFSet %G' % (1.26))
        MSO7034B.write(':CHANnel3:SCALe %G MV' % (20.0))
        MSO7034B.write(':CHANnel3:OFFSet %G' % (2.6125))
        MSO7034B.write(':CHANnel4:SCALe %G' % (10.0))
        MSO7034B.write(':CHANnel4:OFFSet %G' % (10.0))
        EDUX1052G.write(':HARDcopy:INKSaver %d' % (0))
        EDUX1052G.write(':CHANnel1:DISPlay %d' % (1))
        EDUX1052G.write(':CHANnel2:DISPlay %d' % (1))
        EDUX1052G.write(':TRIGger:EDGE:SOURce %s' % ('EXTernal'))
        EDUX1052G.write(':TRIGger:EDGE:LEVel %G' % (0.5))
        EDUX1052G.write(':TRIGger:EDGE:SLOPe %s' % ('POSitive'))
        EDUX1052G.write(':CHANnel1:PROBe %G' % (10.0))
        EDUX1052G.write(':CHANnel2:PROBe %G' % (10.0))
        EDUX1052G.write(':CHANnel1:LABel "%s"' % (labels[4]))
        EDUX1052G.write(':CHANnel2:LABel "%s"' % (labels[5]))
        EDUX1052G.write(':DISPlay:LABel %d' % (1))
        EDUX1052G.write(':CHANnel1:SCALe %G' % (10.0))
        EDUX1052G.write(':CHANnel1:OFFSet %G' % (20.0))
        EDUX1052G.write(':CHANnel2:SCALe %G mV' % (200.0))
        EDUX1052G.write(':TIMebase:SCALe %G US' % (2.0))
    finally:
        print('done')

    # GO!

    # df_scope1 = pd.DataFrame(columns=[
    #     'resistance_ohm', 'frequency_Hz', 'V_SW', 'I_IN', 'I_OUT','I_P'
    # ])
    # df_scope2 = pd.DataFrame(columns=[
    #     'resistance_ohm', 'frequency_Hz', 'V_SEC','V_OUT'
    # ])
    df_scope_measurements = pd.DataFrame(columns=[
        'resistance_ohm', 'frequency_Hz', 'I_P_PP', 'I_P_AMP', 'I_P_AVG',
        'I_OUT_AMP', 'I_OUT_AVG', 'I_IN_AMP', 'I_IN_AVG', 'V_SW_FREQ',
        'V_SW_AMP', 'V_OUT_AVG', 'V_OUT_PP', 'V_SEC_AMP'
    ])
    df_meters = pd.DataFrame(columns=[
        'resistance_ohm', 'frequency_Hz', 'V_IN_V', 'I_IN_A', 'P_IN_W',
        'I_OUT_A', 'V_OUT_V', 'P_OUT_W'
    ])
    # First, we do Section 4.1
    R_LOAD = R_LOADS[0]
    EL34143A.write(':SOURce:RESistance:LEVel:IMMediate:AMPLitude %G' %
                   (R_LOAD))
    
    count = 0
    last = -1
    steps = len(R_LOADS) + len(freqs) -1
    if SHORT:
        for freq in freqs:
            now = time.time()
            count = count + 1
            print(f'Step {count}/{steps}')

            if last != -1:
                dt = now - last
                remaining_time = (steps - count - 1) * dt
                print(f'{time.strftime('%H:%M:%S', time.gmtime(remaining_time))} remains')
            filename = f'4.1_f{freq}r{R_LOAD}'
            v33521A.write(':SOURce:APPLy:SQUare %G,%G,%G' % (freq, 5.0, 2.5))
            # Slowly ramp duty cycle to 50% to prevent overloading E-load
            v33521A.write(':SOURce:FUNCtion:SHAPe:SQUare:DCYCle %G' % (1))
            EL34143A.write(':SOURce:RESistance:LEVel:IMMediate:AMPLitude %G' %
                    (R_LOAD))
            EL34143A.write(':OUTPut:STATe %d' % (1))
            for x in range(1, 51, 2):
                v33521A.write(':SOURce:FUNCtion:SHAPe:SQUare:DCYCle %G' % (x))
                time.sleep(0.2)
            v33521A.write(':SOURce:FUNCtion:SHAPe:SQUare:DCYCle %G' % (50))

            # Wait for system to stabilize
            time.sleep(1)
            # Get measurements
            # Get E-Load, as used for offset
            v_out = EL34143A.query_ascii_values(':MEASure:SCALar:VOLTage:ACDC?')[0]

            p_out = EL34143A.query_ascii_values(':MEASure:SCALar:POWer:DC?')[0]

            i_out = EL34143A.query_ascii_values(':MEASure:SCALar:CURRent:ACDC?')[0]

            EDUX1052G.write(':CHANnel2:OFFSet %G' % (v_out - 0.3))
            MSO7034B.write(':CHANnel3:OFFSet %G' % (i_out))

            # Get DMM, for other offset
            temp_values = EDU34450A.query_ascii_values(':READ?')

            i_in = temp_values[0]
            v_in = temp_values[1]

            MSO7034B.write(':CHANnel2:OFFSet %G' % (i_in))

            # Save DMM Measurements
            new_row = {\
                    'resistance_ohm':R_LOAD,\
                    'frequency_Hz':freq,\
                    'V_IN_V':v_in,\
                    'I_IN_A':i_in,\
                    'P_IN_W':v_in * i_in,\
                    'I_OUT_A':i_out,\
                    'V_OUT_V':v_out,\
                    'P_OUT_W':p_out\
                }
            df_meters = df_meters._append(new_row, ignore_index=True)

            # Scope measurements
            MSO7034B.write(':SINGle')
            EDUX1052G.write(':SINGle')

            MSO7034B.write(':MEASure:CLEar')
            MSO7034B.write(':MEASure:STATistics:RESet')
            i_out_amp = MSO7034B.query_ascii_values(':MEASure:VAMPlitude? %s' %
                                                    ('CHANNEL3'))[0]
            i_out_avg = MSO7034B.query_ascii_values(':MEASure:VAVerage? %s' %
                                                    ('CHANNEL3'))[0]
            i_in_amp = MSO7034B.query_ascii_values(':MEASure:VAMPlitude? %s' %
                                                ('CHANNEL2'))[0]
            i_in_avg = MSO7034B.query_ascii_values(':MEASure:VAVerage? %s' %
                                                ('CHANNEL2'))[0]
            i_p_pp = MSO7034B.query_ascii_values(':MEASure:VPP? %s' %
                                                ('CHANNEL4'))[0]
            i_p_amp = MSO7034B.query_ascii_values(':MEASure:VAMPlitude? %s' %
                                                ('CHANNEL4'))[0]
            i_p_avg = MSO7034B.query_ascii_values(':MEASure:VAVerage? %s' %
                                                ('CHANNEL4'))[0]
            v_sw_freq = MSO7034B.query_ascii_values(':MEASure:FREQuency? %s' %
                                                    ('CHANNEL1'))[0]
            v_sw_amp = MSO7034B.query_ascii_values(':MEASure:VAMPlitude? %s' %
                                                ('CHANNEL1'))[0]

            MSO7034B.write(':MEASure:CLEar')
            v_out_avg = EDUX1052G.query_ascii_values(':MEASure:VAVerage? %s,%s' %
                                                    ('DISPlay', 'CHANnel2'))[0]
            v_out_pp = EDUX1052G.query_ascii_values(':MEASure:VPP? %s' %
                                                    ('CHANnel2'))[0]
            v_sec_amp = EDUX1052G.query_ascii_values(':MEASure:VAMPlitude? %s' %
                                                    ('CHANnel1'))[0]
            EDUX1052G.write(':MEASure:CLEar')

            new_row = {\
                    'resistance_ohm':R_LOAD,\
                    'frequency_Hz':freq,\
                    'I_P_PP':i_p_pp,\
                    'I_P_AMP':i_p_amp,\
                    'I_P_AVG':i_p_avg,\
                    'I_OUT_AMP':i_out_amp,\
                    'I_OUT_AVG':i_out_avg,\
                    'I_IN_AMP':i_in_amp,\
                    'I_IN_AVG':i_in_avg,\
                    'V_SW_FREQ':v_sw_freq,\
                    'V_SW_AMP':v_sw_amp,\
                    'V_OUT_AVG':v_out_avg,\
                    'V_OUT_PP':v_out_pp,\
                    'V_SEC_AMP':v_sec_amp,\
                }
            df_scope_measurements = df_scope_measurements._append(
                new_row, ignore_index=True)

            # Scope captures
            if PLOT:
                plt.figure()
            for c in range(0, 4):
                MSO7034B.write(':WAVeform:SOURce %s' % ('CHANnel%d' % (c + 1)))
                MSO7034B.write(':WAVeform:POINts %s' % ('MAXimum'))
                MSO7034B.write(':WAVeform:FORMat %s' % ('WORD'))
                MSO7034B.write(':WAVeform:UNSigned %d' % (1))
                MSO7034B.write(':WAVeform:BYTeorder %s' % ('LSBFirst'))
                r = MSO7034B.query(':waveform:preamble?')
                xinc, xorg, xref, yinc, yorg, yref = [
                    float(i) for i in r.split(',')[4:]
                ]
                binary_block_data = MSO7034B.query_binary_values(':WAVeform:DATA?',
                                                                datatype='H')
                acq_data = np.array(binary_block_data)
                scaled_data = (acq_data - yref) * yinc + yorg
                if c == 0:
                    times = np.arange(0, xinc * len(acq_data), xinc)
                    dat_tmp = {
                        "Time (s)": times[0:min(len(times), len(scaled_data))],
                        labels[c]: scaled_data[0:min(len(times), len(scaled_data))]
                    }
                    dframe = pd.DataFrame(dat_tmp)
                else:
                    dframe.insert(c + 1, labels[c],
                                scaled_data[0:min(len(times), len(scaled_data))])
                if PLOT:
                    plt.plot(times[0:min(len(times), len(scaled_data))],
                            scaled_data[0:min(len(times), len(scaled_data))])
            dframe.to_csv(p.Path(wavepath, f'{filename}_MSO7034B.csv'),
                        index=False)

            for c in range(0, 2):
                EDUX1052G.write(':WAVeform:SOURce %s' % ('CHANnel%d' % (c + 1)))
                EDUX1052G.write(':WAVeform:POINts %s' % ('MAXimum'))
                EDUX1052G.write(':WAVeform:FORMat %s' % ('WORD'))
                EDUX1052G.write(':WAVeform:UNSigned %d' % (1))
                EDUX1052G.write(':WAVeform:BYTeorder %s' % ('LSBFirst'))
                r = EDUX1052G.query(':waveform:preamble?')
                xinc, xorg, xref, yinc, yorg, yref = [
                    float(i) for i in r.split(',')[4:]
                ]
                binary_block_data = EDUX1052G.query_binary_values(
                    ':WAVeform:DATA?', datatype='H')
                acq_data = np.array(binary_block_data)
                scaled_data = (acq_data - yref) * yinc + yorg
                if c == 0:
                    times = np.arange(0, xinc * len(acq_data), xinc)
                    dat_tmp = {
                        "Time (s)": times[0:min(len(times), len(scaled_data))],
                        labels[c + 4]:
                        scaled_data[0:min(len(times), len(scaled_data))]
                    }
                    dframe = pd.DataFrame(dat_tmp)
                else:
                    dframe.insert(c + 1, labels[c + 4],
                                scaled_data[0:min(len(times), len(scaled_data))])
                if PLOT:
                    plt.plot(times[0:min(len(times), len(scaled_data))],
                            scaled_data[0:min(len(times), len(scaled_data))])
            dframe.to_csv(p.Path(wavepath, f'{filename}_EDUX1052G.csv'),
                        index=False)
            if PLOT:
                if SHOW_PLOT:
                    plt.show()
                plt.savefig(p.Path(pltpath, f'{filename}.svg'), format='svg')
                plt.close()

            img = MSO7034B.query_binary_values(':DISPlay:DATA? %s,%s,%s' %
                                            ('PNG', 'SCReen', 'COLor'),
                                            datatype='c')
            with open(p.Path(sspath, f'{filename}_MSO7034B.png'), 'wb') as f:
                for b in img:
                    f.write(b)
            img = EDUX1052G.query_binary_values(':DISPlay:DATA? %s,%s' %
                                                ('PNG', 'COLor'),
                                                datatype='c')
            with open(p.Path(sspath, f'{filename}_EDUX1052G.png'), 'wb') as f:
                for b in img:
                    f.write(b)
            MSO7034B.write(':RUN')
            EDUX1052G.write(':RUN')
            last = now

        freq = MANUAL_FREQS[1]
        for R_LOAD in R_LOADS[1:]:
            now = time.time()
            count = count + 1
            filename = f'4.2_f{freq}r{R_LOAD}'
            v33521A.write(':SOURce:APPLy:SQUare %G,%G,%G' % (freq, 5.0, 2.5))
            # Slowly ramp duty cycle to 50% to prevent overloading E-load
            v33521A.write(':SOURce:FUNCtion:SHAPe:SQUare:DCYCle %G' % (1))
            EL34143A.write(':SOURce:RESistance:LEVel:IMMediate:AMPLitude %G' %
                    (R_LOAD))
            EL34143A.write(':OUTPut:STATe %d' % (1))
            for x in range(1, 51, 5):
                v33521A.write(':SOURce:FUNCtion:SHAPe:SQUare:DCYCle %G' % (x))
                time.sleep(0.2)
            v33521A.write(':SOURce:FUNCtion:SHAPe:SQUare:DCYCle %G' % (50))

            print(f'Step {count}/{steps}')

            if last != -1:
                dt = now - last
                remaining_time = (steps - count - 1) * dt
                print(f'{time.strftime('%H:%M:%S', time.gmtime(remaining_time))} remains')

            # Wait for system to stabilize
            time.sleep(1)
            # Get measurements
            # Get E-Load, as used for offset
            v_out = EL34143A.query_ascii_values(':MEASure:SCALar:VOLTage:ACDC?')[0]

            p_out = EL34143A.query_ascii_values(':MEASure:SCALar:POWer:DC?')[0]

            i_out = EL34143A.query_ascii_values(':MEASure:SCALar:CURRent:ACDC?')[0]

            EDUX1052G.write(':CHANnel2:OFFSet %G' % (v_out - 0.3))
            MSO7034B.write(':CHANnel3:OFFSet %G' % (i_out))

            # Get DMM, for other offset
            temp_values = EDU34450A.query_ascii_values(':READ?')

            i_in = temp_values[0]
            v_in = temp_values[1]

            MSO7034B.write(':CHANnel2:OFFSet %G' % (i_in))

            # Save DMM Measurements
            new_row = {\
                    'resistance_ohm':R_LOAD,\
                    'frequency_Hz':freq,\
                    'V_IN_V':v_in,\
                    'I_IN_A':i_in,\
                    'P_IN_W':v_in * i_in,\
                    'I_OUT_A':i_out,\
                    'V_OUT_V':v_out,\
                    'P_OUT_W':p_out\
                }
            df_meters = df_meters._append(new_row, ignore_index=True)

            # Scope measurements
            MSO7034B.write(':SINGle')
            EDUX1052G.write(':SINGle')

            MSO7034B.write(':MEASure:CLEar')
            MSO7034B.write(':MEASure:STATistics:RESet')
            i_out_amp = MSO7034B.query_ascii_values(':MEASure:VAMPlitude? %s' %
                                                    ('CHANNEL3'))[0]
            i_out_avg = MSO7034B.query_ascii_values(':MEASure:VAVerage? %s' %
                                                    ('CHANNEL3'))[0]
            i_in_amp = MSO7034B.query_ascii_values(':MEASure:VAMPlitude? %s' %
                                                ('CHANNEL2'))[0]
            i_in_avg = MSO7034B.query_ascii_values(':MEASure:VAVerage? %s' %
                                                ('CHANNEL2'))[0]
            i_p_pp = MSO7034B.query_ascii_values(':MEASure:VPP? %s' %
                                                ('CHANNEL4'))[0]
            i_p_amp = MSO7034B.query_ascii_values(':MEASure:VAMPlitude? %s' %
                                                ('CHANNEL4'))[0]
            i_p_avg = MSO7034B.query_ascii_values(':MEASure:VAVerage? %s' %
                                                ('CHANNEL4'))[0]
            v_sw_freq = MSO7034B.query_ascii_values(':MEASure:FREQuency? %s' %
                                                    ('CHANNEL1'))[0]
            v_sw_amp = MSO7034B.query_ascii_values(':MEASure:VAMPlitude? %s' %
                                                ('CHANNEL1'))[0]

            MSO7034B.write(':MEASure:CLEar')
            v_out_avg = EDUX1052G.query_ascii_values(':MEASure:VAVerage? %s,%s' %
                                                    ('DISPlay', 'CHANnel2'))[0]
            v_out_pp = EDUX1052G.query_ascii_values(':MEASure:VPP? %s' %
                                                    ('CHANnel2'))[0]
            v_sec_amp = EDUX1052G.query_ascii_values(':MEASure:VAMPlitude? %s' %
                                                    ('CHANnel1'))[0]
            EDUX1052G.write(':MEASure:CLEar')

            new_row = {\
                    'resistance_ohm':R_LOAD,\
                    'frequency_Hz':freq,\
                    'I_P_PP':i_p_pp,\
                    'I_P_AMP':i_p_amp,\
                    'I_P_AVG':i_p_avg,\
                    'I_OUT_AMP':i_out_amp,\
                    'I_OUT_AVG':i_out_avg,\
                    'I_IN_AMP':i_in_amp,\
                    'I_IN_AVG':i_in_avg,\
                    'V_SW_FREQ':v_sw_freq,\
                    'V_SW_AMP':v_sw_amp,\
                    'V_OUT_AVG':v_out_avg,\
                    'V_OUT_PP':v_out_pp,\
                    'V_SEC_AMP':v_sec_amp,\
                }
            df_scope_measurements = df_scope_measurements._append(
                new_row, ignore_index=True)

            # Scope captures
            if PLOT:
                plt.figure()
            for c in range(0, 4):
                MSO7034B.write(':WAVeform:SOURce %s' % ('CHANnel%d' % (c + 1)))
                MSO7034B.write(':WAVeform:POINts %s' % ('MAXimum'))
                MSO7034B.write(':WAVeform:FORMat %s' % ('WORD'))
                MSO7034B.write(':WAVeform:UNSigned %d' % (1))
                MSO7034B.write(':WAVeform:BYTeorder %s' % ('LSBFirst'))
                r = MSO7034B.query(':waveform:preamble?')
                xinc, xorg, xref, yinc, yorg, yref = [
                    float(i) for i in r.split(',')[4:]
                ]
                binary_block_data = MSO7034B.query_binary_values(':WAVeform:DATA?',
                                                                datatype='H')
                acq_data = np.array(binary_block_data)
                scaled_data = (acq_data - yref) * yinc + yorg
                if c == 0:
                    times = np.arange(0, xinc * len(acq_data), xinc)
                    dat_tmp = {
                        "Time (s)": times[0:min(len(times), len(scaled_data))],
                        labels[c]: scaled_data[0:min(len(times), len(scaled_data))]
                    }
                    dframe = pd.DataFrame(dat_tmp)
                else:
                    dframe.insert(c + 1, labels[c],
                                scaled_data[0:min(len(times), len(scaled_data))])
                if PLOT:
                    plt.plot(times[0:min(len(times), len(scaled_data))],
                            scaled_data[0:min(len(times), len(scaled_data))])
            dframe.to_csv(p.Path(wavepath, f'{filename}_MSO7034B.csv'),
                        index=False)

            for c in range(0, 2):
                EDUX1052G.write(':WAVeform:SOURce %s' % ('CHANnel%d' % (c + 1)))
                EDUX1052G.write(':WAVeform:POINts %s' % ('MAXimum'))
                EDUX1052G.write(':WAVeform:FORMat %s' % ('WORD'))
                EDUX1052G.write(':WAVeform:UNSigned %d' % (1))
                EDUX1052G.write(':WAVeform:BYTeorder %s' % ('LSBFirst'))
                r = EDUX1052G.query(':waveform:preamble?')
                xinc, xorg, xref, yinc, yorg, yref = [
                    float(i) for i in r.split(',')[4:]
                ]
                binary_block_data = EDUX1052G.query_binary_values(
                    ':WAVeform:DATA?', datatype='H')
                acq_data = np.array(binary_block_data)
                scaled_data = (acq_data - yref) * yinc + yorg
                if c == 0:
                    times = np.arange(0, xinc * len(acq_data), xinc)
                    dat_tmp = {
                        "Time (s)": times[0:min(len(times), len(scaled_data))],
                        labels[c + 4]:
                        scaled_data[0:min(len(times), len(scaled_data))]
                    }
                    dframe = pd.DataFrame(dat_tmp)
                else:
                    dframe.insert(c + 1, labels[c + 4],
                                scaled_data[0:min(len(times), len(scaled_data))])
                if PLOT:
                    plt.plot(times[0:min(len(times), len(scaled_data))],
                            scaled_data[0:min(len(times), len(scaled_data))])
            dframe.to_csv(p.Path(wavepath, f'{filename}_EDUX1052G.csv'),
                        index=False)
            if PLOT:
                if SHOW_PLOT:
                    plt.show()
                plt.savefig(p.Path(pltpath, f'{filename}.svg'), format='svg')
                plt.close()

            img = MSO7034B.query_binary_values(':DISPlay:DATA? %s,%s,%s' %
                                            ('PNG', 'SCReen', 'COLor'),
                                            datatype='c')
            with open(p.Path(sspath, f'{filename}_MSO7034B.png'), 'wb') as f:
                for b in img:
                    f.write(b)
            img = EDUX1052G.query_binary_values(':DISPlay:DATA? %s,%s' %
                                                ('PNG', 'COLor'),
                                                datatype='c')
            with open(p.Path(sspath, f'{filename}_EDUX1052G.png'), 'wb') as f:
                for b in img:
                    f.write(b)
            MSO7034B.write(':RUN')
            EDUX1052G.write(':RUN')
            last = now
        df_scope_measurements.to_csv(p.Path(dir, 'data_scope_1.csv'), index=False)
        df_meters.to_csv(p.Path(dir, 'data_meter_1.csv'), index=False)
    
    if LONG:
        df_scope_measurements = pd.DataFrame(columns=[
            'resistance_ohm', 'frequency_Hz', 'I_P_PP', 'I_P_AMP', 'I_P_AVG',
            'I_OUT_AMP', 'I_OUT_AVG', 'I_IN_AMP', 'I_IN_AVG', 'V_SW_FREQ',
            'V_SW_AMP', 'V_OUT_AVG', 'V_OUT_PP', 'V_SEC_AMP'
        ])
        df_meters = pd.DataFrame(columns=[
            'resistance_ohm', 'frequency_Hz', 'V_IN_V', 'I_IN_A', 'P_IN_W',
            'I_OUT_A', 'V_OUT_V', 'P_OUT_W'
        ])
        steps = len(R_LOADS) * len(freqs) -1

        for R_LOAD in R_LOADS:
            for freq in freqs:
                for R_LOAD in R_LOADS:
                    now = time.time()
                    count = count + 1
                    filename = f'f{freq}r{R_LOAD}'
                    v33521A.write(':SOURce:APPLy:SQUare %G,%G,%G' % (freq, 5.0, 2.5))
                    # Slowly ramp duty cycle to 50% to prevent overloading E-load
                    v33521A.write(':SOURce:FUNCtion:SHAPe:SQUare:DCYCle %G' % (1))
                    EL34143A.write(':SOURce:RESistance:LEVel:IMMediate:AMPLitude %G' %
                            (R_LOAD))
                    EL34143A.write(':OUTPut:STATe %d' % (1))
                    for x in range(1, 51, 5):
                        v33521A.write(':SOURce:FUNCtion:SHAPe:SQUare:DCYCle %G' % (x))
                        time.sleep(0.2)
                    v33521A.write(':SOURce:FUNCtion:SHAPe:SQUare:DCYCle %G' % (50))

                    print(f'Step {count}/{steps}')

                    if last != -1:
                        dt = now - last
                        remaining_time = (steps - count - 1) * dt
                        print(f'{time.strftime('%H:%M:%S', time.gmtime(remaining_time))} remains')

                    # Wait for system to stabilize
                    # time.sleep(0.25)
                    # Get measurements
                    # Get E-Load, as used for offset
                    v_out = EL34143A.query_ascii_values(':MEASure:SCALar:VOLTage:ACDC?')[0]

                    p_out = EL34143A.query_ascii_values(':MEASure:SCALar:POWer:DC?')[0]

                    i_out = EL34143A.query_ascii_values(':MEASure:SCALar:CURRent:ACDC?')[0]

                    EDUX1052G.write(':CHANnel2:OFFSet %G' % (v_out - 0.3))
                    MSO7034B.write(':CHANnel3:OFFSet %G' % (i_out))

                    # Get DMM, for other offset
                    temp_values = EDU34450A.query_ascii_values(':READ?')

                    i_in = temp_values[0]
                    v_in = temp_values[1]

                    MSO7034B.write(':CHANnel2:OFFSet %G' % (i_in))

                    # Save DMM Measurements
                    new_row = {\
                            'resistance_ohm':R_LOAD,\
                            'frequency_Hz':freq,\
                            'V_IN_V':v_in,\
                            'I_IN_A':i_in,\
                            'P_IN_W':v_in * i_in,\
                            'I_OUT_A':i_out,\
                            'V_OUT_V':v_out,\
                            'P_OUT_W':p_out\
                        }
                    df_meters = df_meters._append(new_row, ignore_index=True)

                    # Scope measurements
                    MSO7034B.write(':AUToscale')
                    EDUX1052G.write(':AUToscale')
                    time.sleep(0.5)
                    MSO7034B.write(':SINGle')
                    EDUX1052G.write(':SINGle')

                    MSO7034B.write(':MEASure:CLEar')
                    MSO7034B.write(':MEASure:STATistics:RESet')
                    i_out_amp = MSO7034B.query_ascii_values(':MEASure:VAMPlitude? %s' %
                                                            ('CHANNEL3'))[0]
                    i_out_avg = MSO7034B.query_ascii_values(':MEASure:VAVerage? %s' %
                                                            ('CHANNEL3'))[0]
                    i_in_amp = MSO7034B.query_ascii_values(':MEASure:VAMPlitude? %s' %
                                                        ('CHANNEL2'))[0]
                    i_in_avg = MSO7034B.query_ascii_values(':MEASure:VAVerage? %s' %
                                                        ('CHANNEL2'))[0]
                    i_p_pp = MSO7034B.query_ascii_values(':MEASure:VPP? %s' %
                                                        ('CHANNEL4'))[0]
                    i_p_amp = MSO7034B.query_ascii_values(':MEASure:VAMPlitude? %s' %
                                                        ('CHANNEL4'))[0]
                    i_p_avg = MSO7034B.query_ascii_values(':MEASure:VAVerage? %s' %
                                                        ('CHANNEL4'))[0]
                    v_sw_freq = MSO7034B.query_ascii_values(':MEASure:FREQuency? %s' %
                                                            ('CHANNEL1'))[0]
                    v_sw_amp = MSO7034B.query_ascii_values(':MEASure:VAMPlitude? %s' %
                                                        ('CHANNEL1'))[0]

                    MSO7034B.write(':MEASure:CLEar')
                    v_out_avg = EDUX1052G.query_ascii_values(':MEASure:VAVerage? %s,%s' %
                                                            ('DISPlay', 'CHANnel2'))[0]
                    v_out_pp = EDUX1052G.query_ascii_values(':MEASure:VPP? %s' %
                                                            ('CHANnel2'))[0]
                    v_sec_amp = EDUX1052G.query_ascii_values(':MEASure:VAMPlitude? %s' %
                                                            ('CHANnel1'))[0]
                    EDUX1052G.write(':MEASure:CLEar')

                    new_row = {\
                            'resistance_ohm':R_LOAD,\
                            'frequency_Hz':freq,\
                            'I_P_PP':i_p_pp,\
                            'I_P_AMP':i_p_amp,\
                            'I_P_AVG':i_p_avg,\
                            'I_OUT_AMP':i_out_amp,\
                            'I_OUT_AVG':i_out_avg,\
                            'I_IN_AMP':i_in_amp,\
                            'I_IN_AVG':i_in_avg,\
                            'V_SW_FREQ':v_sw_freq,\
                            'V_SW_AMP':v_sw_amp,\
                            'V_OUT_AVG':v_out_avg,\
                            'V_OUT_PP':v_out_pp,\
                            'V_SEC_AMP':v_sec_amp,\
                        }
                    df_scope_measurements = df_scope_measurements._append(
                        new_row, ignore_index=True)

                    # Scope captures
                    if PLOT:
                        plt.figure()
                    for c in range(0, 4):
                        MSO7034B.write(':WAVeform:SOURce %s' % ('CHANnel%d' % (c + 1)))
                        MSO7034B.write(':WAVeform:POINts %s' % ('MAXimum'))
                        MSO7034B.write(':WAVeform:FORMat %s' % ('WORD'))
                        MSO7034B.write(':WAVeform:UNSigned %d' % (1))
                        MSO7034B.write(':WAVeform:BYTeorder %s' % ('LSBFirst'))
                        r = MSO7034B.query(':waveform:preamble?')
                        xinc, xorg, xref, yinc, yorg, yref = [
                            float(i) for i in r.split(',')[4:]
                        ]
                        binary_block_data = MSO7034B.query_binary_values(':WAVeform:DATA?',
                                                                        datatype='H')
                        acq_data = np.array(binary_block_data)
                        scaled_data = (acq_data - yref) * yinc + yorg
                        if c == 0:
                            times = np.arange(0, xinc * len(acq_data), xinc)
                            dat_tmp = {
                                "Time (s)": times[0:min(len(times), len(scaled_data))],
                                labels[c]: scaled_data[0:min(len(times), len(scaled_data))]
                            }
                            dframe = pd.DataFrame(dat_tmp)
                        else:
                            dframe.insert(c + 1, labels[c],
                                        scaled_data[0:min(len(times), len(scaled_data))])
                        if PLOT:
                            plt.plot(times[0:min(len(times), len(scaled_data))],
                                    scaled_data[0:min(len(times), len(scaled_data))])
                    dframe.to_csv(p.Path(wavepath, f'{filename}_MSO7034B.csv'),
                                index=False)

                    for c in range(0, 2):
                        EDUX1052G.write(':WAVeform:SOURce %s' % ('CHANnel%d' % (c + 1)))
                        EDUX1052G.write(':WAVeform:POINts %s' % ('MAXimum'))
                        EDUX1052G.write(':WAVeform:FORMat %s' % ('WORD'))
                        EDUX1052G.write(':WAVeform:UNSigned %d' % (1))
                        EDUX1052G.write(':WAVeform:BYTeorder %s' % ('LSBFirst'))
                        r = EDUX1052G.query(':waveform:preamble?')
                        xinc, xorg, xref, yinc, yorg, yref = [
                            float(i) for i in r.split(',')[4:]
                        ]
                        binary_block_data = EDUX1052G.query_binary_values(
                            ':WAVeform:DATA?', datatype='H')
                        acq_data = np.array(binary_block_data)
                        scaled_data = (acq_data - yref) * yinc + yorg
                        if c == 0:
                            times = np.arange(0, xinc * len(acq_data), xinc)
                            dat_tmp = {
                                "Time (s)": times[0:min(len(times), len(scaled_data))],
                                labels[c + 4]:
                                scaled_data[0:min(len(times), len(scaled_data))]
                            }
                            dframe = pd.DataFrame(dat_tmp)
                        else:
                            dframe.insert(c + 1, labels[c + 4],
                                        scaled_data[0:min(len(times), len(scaled_data))])
                        if PLOT:
                            plt.plot(times[0:min(len(times), len(scaled_data))],
                                    scaled_data[0:min(len(times), len(scaled_data))])
                    dframe.to_csv(p.Path(wavepath, f'{filename}_EDUX1052G.csv'),
                                index=False)
                    if PLOT:
                        if SHOW_PLOT:
                            plt.show()
                        plt.savefig(p.Path(pltpath, f'{filename}.svg'), format='svg')
                        plt.close()

                    img = MSO7034B.query_binary_values(':DISPlay:DATA? %s,%s,%s' %
                                                    ('PNG', 'SCReen', 'COLor'),
                                                    datatype='c')
                    with open(p.Path(sspath, f'{filename}_MSO7034B.png'), 'wb') as f:
                        for b in img:
                            f.write(b)
                    img = EDUX1052G.query_binary_values(':DISPlay:DATA? %s,%s' %
                                                        ('PNG', 'COLor'),
                                                        datatype='c')
                    with open(p.Path(sspath, f'{filename}_EDUX1052G.png'), 'wb') as f:
                        for b in img:
                            f.write(b)
                    MSO7034B.write(':RUN')
                    EDUX1052G.write(':RUN')
                    last = now
        

        df_scope_measurements.to_csv(p.Path(dir, 'data_scope_2.csv'), index=False)
        df_meters.to_csv(p.Path(dir, 'data_meter_2.csv'), index=False)
except BaseException as e:
    print(e)
finally:
    print("Cleaning up and Exiting...")
    v33521A.write(':OUTPut:STATe %d' % (0))
    time.sleep(0.5)
    EL34143A.write(':OUTPut:STATe %d' % (0))
    time.sleep(2)
    v33521A.close()
    EL34143A.close()
    MSO7034B.close()
    EDU34450A.close()
    EDUX1052G.close()
    time.sleep(1)
    rm.close()
    exit()

# end of Experiment
