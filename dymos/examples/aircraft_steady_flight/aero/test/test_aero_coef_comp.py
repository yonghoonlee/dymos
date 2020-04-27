import unittest

import numpy as np
from numpy import array

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from dymos.examples.aircraft_steady_flight.aero.aero_coef_comp import AeroCoefComp


class TestAeroCoefComp(unittest.TestCase):

    def test_aero_coefs(self):

        nn = 100

        prob = om.Problem(model=om.Group())

        prob.model.add_subsystem(name='aero', subsys=AeroCoefComp(vec_size=nn, method='lagrange3'))

        ivc = prob.model.add_subsystem(name='ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('mach', val=np.zeros(nn), units=None)
        ivc.add_output('alpha', val=np.zeros(nn), units='rad')
        ivc.add_output('alt', val=np.zeros(nn), units='km')
        ivc.add_output('eta', val=np.zeros(nn), units='rad')

        prob.model.connect('mach', 'aero.mach')
        prob.model.connect('alt', 'aero.alt')
        prob.model.connect('alpha', 'aero.alpha')
        prob.model.connect('eta', 'aero.eta')

        prob.setup()

        # Data from mission allocation test_mission.py

        prob['mach'] = 0.85*np.ones(nn)

        prob['alt'] = array([0.000000000000,   0.553093433783,   1.129059821455,
                             1.730321131057,   2.359470472370,   3.019070277784,
                             3.711168239580,   4.436240405573,   5.191067159896,
                             5.964994808747,   6.735049468246,   7.470889546615,
                             8.161630727529,   8.802979718921,   9.392586590598,
                             9.931246639900,  10.423571988020,  10.873121665339,
                             11.278254552329,  11.636886841861,  11.946934726808,
                             12.206320941201,  12.415521693453,  12.581431037308,
                             12.711935344453,  12.814920986570,  12.898274335344,
                             12.968850492916,  13.028672695806,  13.078350251994,
                             13.118492292627,  13.149707948854,  13.172677881382,
                             13.188733424875,  13.199553239017,  13.206819241941,
                             13.212213351786,  13.217370981798,  13.222979357103,
                             13.228840435808,  13.234722273961,  13.240392927606,
                             13.245621713804,  13.250264124574,  13.254315647580,
                             13.257784596604,  13.260679285430,  13.263008105325,
                             13.264868833986,  13.266620793272,  13.268670889348,
                             13.271426028378,  13.275293116524,  13.280509793866,
                             13.286383339939,  13.291903070367,  13.296058025153,
                             13.297837244300,  13.296398451540,  13.292740533937,
                             13.288995715167,  13.287312803195,  13.289840605987,
                             13.298604587026,  13.312409379072,  13.326609860865,
                             13.336391714322,  13.336940621360,  13.323487511639,
                             13.296086362331,  13.263826582079,  13.236790858000,
                             13.225061877209,  13.238721274816,  13.279438840050,
                             13.320138278003,  13.327607990471,  13.268636379249,
                             13.110011846132,  12.823515993940,  12.413224099419,
                             11.896060154844,  11.288981066656,  10.608943741299,
                             9.872630230147,   9.093575118885,   8.284188848025,
                             7.456873953300,   6.622384413067,   5.789034047114,
                             4.969148590717,   4.192948026585,   3.478203759977,
                             2.828278733006,   2.239515506987,   1.705768191211,
                             1.220548755122,   0.777873316704,   0.372512860834,
                             0.])

        prob['alpha'] = array([-0.038156436870, -0.037134586723, -0.035997353057, -0.034718312665,
                               -0.033265200365, -0.031598423872, -0.029670064208, -0.027424901341,
                               -0.024807249082, -0.021781079351, -0.018371926465, -0.014683657235,
                               -0.010780801655, -0.006721822873, -0.002572325541,  0.001611091821,
                               0.005799529666,  0.009962368108,  0.013999160174,  0.017789495537,
                               0.021235517833,  0.024244757396,  0.026755586103,  0.028798241839,
                               0.030434448784,  0.031740936964,  0.032805536919,  0.033710907378,
                               0.034479881187,  0.035116985330,  0.035627601932,  0.036017872241,
                               0.036295521599,  0.036477720544,  0.036587059227,  0.036646511540,
                               0.036679225352,  0.036707829010,  0.036742095579,  0.036779375923,
                               0.036816513644,  0.036850347463,  0.036877727136,  0.036896665434,
                               0.036907132050,  0.036909285357,  0.036903286046,  0.036889297681,
                               0.036868683206,  0.036846433488,  0.036828216628,  0.036819696070,
                               0.036826534790,  0.036852158161,  0.036887187320,  0.036917709303,
                               0.036929788846,  0.036909490398,  0.036845121215,  0.036750236576,
                               0.036654335700,  0.036587050685,  0.036577934780,  0.036655043839,
                               0.036803048209,  0.036958282839,  0.037054191743,  0.037023989379,
                               0.036801846092,  0.036388008217,  0.035908639619,  0.035501923793,
                               0.035303539582,  0.035448690186,  0.035966826238,  0.036496415736,
                               0.036579404359,  0.035757839973,  0.033606178143,  0.029847678046,
                               0.024737121268,  0.018734102871,  0.012269948998,  0.005732282405,
                               -0.000503164636, -0.006258471233, -0.011453872007, -0.016060111279,
                               -0.020091449510, -0.023591240399, -0.026596275618, -0.029098355476,
                               -0.031147144368, -0.032824087613, -0.034207606489, -0.035362069690,
                               -0.036337272667, -0.037171028169, -0.037892055217, -0.038522377318])

        prob['eta'] = array([0.070107251814,  0.069284630974,  0.068369433015,  0.067340351212,
                             0.066171359282,  0.064830504261,  0.063279084586,  0.061472451545,
                             0.059365483564,  0.056928774446,  0.054182425445,  0.051209702610,
                             0.048062332911,  0.044787284946,  0.041437414516,  0.038058426361,
                             0.034673733947,  0.031308182371,  0.028043196532,  0.024976478867,
                             0.022187503902,  0.019751413821,  0.017718394516,  0.016064198626,
                             0.014739002540,  0.013680764335,  0.012818402108,  0.012084988686,
                             0.011462045807,  0.010945921633,  0.010532267802,  0.010216117628,
                             0.009991218167,  0.009843660809,  0.009755143045,  0.009707050010,
                             0.009680625138,  0.009657532440,  0.009629852147,  0.009599730702,
                             0.009569725669,  0.009542398587,  0.009520302075,  0.009505046398,
                             0.009496656058,  0.009495002636,  0.009499955825,  0.009511382983,
                             0.009528179662,  0.009546301642,  0.009561155908,  0.009568153245,
                             0.009562704826,  0.009542034755,  0.009513742418,  0.009489101524,
                             0.009479403911,  0.009495941639,  0.009548189470,  0.009625163733,
                             0.009702960815,  0.009757569915,  0.009765044473,  0.009702650720,
                             0.009582807554,  0.009457101635,  0.009379464100,  0.009404012962,
                             0.009584089870,  0.009919492849,  0.010307986460,  0.010637604616,
                             0.010798421264,  0.010680897257,  0.010261152913,  0.009832097802,
                             0.009764892785,  0.010430609174,  0.012173908400,  0.015218519283,
                             0.019357140337,  0.024216479181,  0.029446404456,  0.034732622113,
                             0.039770958850,  0.044417966203,  0.048609891569,  0.052323915726,
                             0.055572345240,  0.058390863032,  0.060809762122,  0.062823033527,
                             0.064471116583,  0.065819853594,  0.066932514089,  0.067860972826,
                             0.068645329742,  0.069316017853,  0.069896140873,  0.070403409938])

        CL_test = array([0.082888007941,  0.089072273825,  0.095950354327,  0.103682321645,
                         0.112464008649,  0.122536141064,  0.134190540205,  0.147763949092,
                         0.163596192996,  0.181906492391,  0.202535196124,  0.224838479909,
                         0.248416262579,  0.272909718715,  0.297919921211,  0.323108403973,
                         0.348311958890,  0.373352486132,  0.397611158711,  0.420360288332,
                         0.441014275330,  0.459006804154,  0.473964204148,  0.486079517294,
                         0.495739175302,  0.503420882965,  0.509669375565,  0.514994819199,
                         0.519530760226,  0.523294962713,  0.526309802799,  0.528601851851,
                         0.530208041979,  0.531232159951,  0.531816228699,  0.532104623496,
                         0.532242705667,  0.532372416915,  0.532553454935,  0.532766080042,
                         0.532987310682,  0.533194131413,  0.533363601788,  0.533480965261,
                         0.533545241461,  0.533556777218,  0.533515934826,  0.533423096413,
                         0.533286786999,  0.533140163667,  0.533020979605,  0.532966954135,
                         0.533015804273,  0.533190398915,  0.533428589588,  0.533638103724,
                         0.533726503050,  0.533601340101,  0.533185004684,  0.532566768248,
                         0.531941254289,  0.531503894925,  0.531449557199,  0.531963339089,
                         0.532945850963,  0.533980470574,  0.534631303775,  0.534460721015,
                         0.533037471536,  0.530360275540,  0.527246038416,  0.524588564851,
                         0.523263927971,  0.524144457022,  0.527439910054,  0.530904941796,
                         0.531683444849,  0.526912861133,  0.513929844291,  0.491013298304,
                         0.459811576691,  0.423253110307,  0.384055408144,  0.344646198247,
                         0.307305101113,  0.273011334595,  0.242149002046,  0.214826811759,
                         0.190919904642,  0.170150003509,  0.152289579601,  0.137386128378,
                         0.125150926630,  0.115107890533,  0.106797459579,  0.099841677003,
                         0.093947633640,  0.088892519528,  0.084506805811,  0.080660230193])

        CD_test = array([0.018719755341,  0.019084861059,  0.019484369873,  0.019924769703,
                         0.020414530547,  0.020964730658,  0.021589747796,  0.022307684226,
                         0.023139502741,  0.024104401596,  0.025207445242,  0.026431747818,
                         0.027774390044,  0.029233245277,  0.030799970794,  0.032464706963,
                         0.034224362433,  0.036071018533,  0.037960750891,  0.039829718286,
                         0.041610298654,  0.043227841304,  0.044620868488,  0.045782269639,
                         0.046730112026,  0.047498244171,  0.048132872861,  0.048680855584,
                         0.049152931163,  0.049548760284,  0.049868899689,  0.050114747444,
                         0.050289181905,  0.050402531968,  0.050469492344,  0.050505228635,
                         0.050525126035,  0.050544248425,  0.050568907260,  0.050596998270,
                         0.050626073473,  0.050653676508,  0.050677354674,  0.050695528265,
                         0.050708083149,  0.050715047056,  0.050716451101,  0.050712330267,
                         0.050703594855,  0.050693779763,  0.050686903899,  0.050686980559,
                         0.050698024424,  0.050722461706,  0.050753652490,  0.050781748996,
                         0.050796867912,  0.050789121423,  0.050750250878,  0.050689883032,
                         0.050628803755,  0.050587796409,  0.050587535895,  0.050647704667,
                         0.050757761458,  0.050873405064,  0.050947970196,  0.050934423630,
                         0.050786795038,  0.050506292519,  0.050181222267,  0.049906195350,
                         0.049772182484,  0.049869649571,  0.050221355953,  0.050592252560,
                         0.050677569394,  0.050173332992,  0.048822063470,  0.046521043459,
                         0.043557702548,  0.040326486386,  0.037139810654,  0.034209230759,
                         0.031657207100,  0.029487110138,  0.027659463852,  0.026124355281,
                         0.024829865012,  0.023728286165,  0.022786107350,  0.021994059861,
                         0.021332763445,  0.020777569791,  0.020306416459,  0.019901757560,
                         0.019550206447,  0.019241644947,  0.018968388593,  0.018724538140])

        prob.run_model()

        assert_near_equal(prob['aero.CL'], CL_test, tolerance=0.005)
        assert_near_equal(prob['aero.CD'], CD_test, tolerance=0.005)
