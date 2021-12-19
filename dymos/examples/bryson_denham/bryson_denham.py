import openmdao.api as om
import dymos as dm
import numpy as np
from matplotlib import pyplot as plt

class BrysonDenhamODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('x', val=np.zeros(nn), desc='position', units='m')
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')
        self.add_input('u', val=np.zeros(nn), desc='acceleration', units='m/s**2')
        self.add_input('L', val=np.zeros(nn), desc='Lagrange term integrated', units='m**2/s**3')

        self.add_output('xdot', val=np.zeros(nn), desc='velocity in x', units='m/s')
        self.add_output('vdot', val=np.zeros(nn), desc='accleration in v', units='m/s/s')
        self.add_output('Ldot', val=np.zeros(nn), desc='Lagrange term integrand', units='m**2/s**4')
        self.declare_coloring(wrt='*', method='fd')

    def compute(self, inputs, outputs):
        x = inputs['x']
        v = inputs['v']
        u = inputs['u']
        L = inputs['L']
        outputs['xdot'] = v
        outputs['vdot'] = u
        outputs['Ldot'] = 0.5*u**2

def BrysonDenhamProblem(
            transcription='radau-ps',
            num_segments=3,
            transcription_order=5,
            optimizer='SLSQP',
            use_pyoptsparse=False,
            sim_times_per_seg=30,
        ):
    p = om.Problem(model=om.Group())

    if not use_pyoptsparse:
        p.driver = om.ScipyOptimizeDriver()
    else:
        p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    if use_pyoptsparse:
        if optimizer == 'SNOPT':
            p.driver.opt_settings['iSumm'] = 6 # show detailed SNOPT output
        elif optimizer == 'IPOPT':
            p.driver.opt_settings['print_level'] = 4
    p.driver.declare_coloring()

    traj = dm.Trajectory()
    p.model.add_subsystem('traj', subsys=traj)

    if transcription == 'gauss-lobatto':
        t = dm.GaussLobatto(
            num_segments=num_segments,
            order=transcription_order,
            compressed=True
        )
    elif transcription == 'radau-ps':
        t = dm.Radau(
            num_segments=num_segments,
            order=transcription_order,
            compressed=True
        )
    phase0 = dm.Phase(ode_class=BrysonDenhamODE, transcription=t)
    traj.add_phase(name='phase0', phase=phase0)
    phase0.set_time_options(fix_initial=True, fix_duration=True, duration_val=1.0, units='s')
    phase0.add_state('x', fix_initial=True, fix_final=True, lower=0.0, upper=2.0, rate_source='xdot', units='m', targets='x')
    phase0.add_state('v', fix_initial=True, fix_final=True, lower=-2.0, upper=2.0, rate_source='vdot', units='m/s', targets='v')
    phase0.add_state('L', fix_initial=True, fix_final=False, lower=-10.0, upper=10.0, rate_source='Ldot', units='m**2/s**3', targets='L')
    phase0.add_control('u', lower=-10.0, upper=10.0, units='m/s**2', continuity=True, rate_continuity=False, targets='u')
    phase0.add_path_constraint('x', lower=0.0, upper=1.0/9.0)
    phase0.add_objective('L', loc='final')

    p.setup()
    
    # initial conditions
    p.set_val('traj.phase0.t_initial', 0.0)
    p.set_val('traj.phase0.t_duration', 1.0)
    p.set_val('traj.phase0.states:x', phase0.interp('x', [0.0, 0.0]))
    p.set_val('traj.phase0.states:v', phase0.interp('v', [1.0, -1.0]))
    p.set_val('traj.phase0.states:L', phase0.interp('L', [0.0, 1.0]))
    
    p.run_driver()
    sim_out = traj.simulate(times_per_seg=sim_times_per_seg)

    return p, sim_out


if __name__ == '__main__':
    # Run Bryson-Denham Problem
    p, sim_out = BrysonDenhamProblem(
        transcription='radau-ps',
        num_segments=6,
        transcription_order=5,
        optimizer='IPOPT', # 'SLSQP', 'IPOPT'
        use_pyoptsparse=True,
        sim_times_per_seg=30,
    )

    # Retrieve solution and simulation results
    t_sol = p.get_val('traj.phase0.timeseries.time')
    x_sol = p.get_val('traj.phase0.timeseries.states:x')
    v_sol = p.get_val('traj.phase0.timeseries.states:v')
    L_sol = p.get_val('traj.phase0.timeseries.states:L')
    u_sol = p.get_val('traj.phase0.timeseries.controls:u')
    t_sim = sim_out.get_val('traj.phase0.timeseries.time')
    x_sim = sim_out.get_val('traj.phase0.timeseries.states:x')
    v_sim = sim_out.get_val('traj.phase0.timeseries.states:v')
    L_sim = sim_out.get_val('traj.phase0.timeseries.states:L')
    u_sim = sim_out.get_val('traj.phase0.timeseries.controls:u')
    
    # Plot
    plt.figure()
    plt.plot(t_sim, x_sim, 'k-')
    plt.plot(t_sol, x_sol, 'ko')
    plt.legend(['simulation result', 'optimal control solution'])
    plt.xlabel('$t$')
    plt.ylabel('$x(t)$')

    plt.figure()
    plt.plot(t_sim, v_sim, 'r-')
    plt.plot(t_sol, v_sol, 'ro')
    plt.legend(['simulation result', 'optimal control solution'])
    plt.xlabel('$t$')
    plt.ylabel('$v(t)=dx/dt$')

    plt.figure()
    plt.plot(t_sim, u_sim, 'b-')
    plt.plot(t_sol, u_sol, 'bo')
    plt.legend(['simulation result', 'optimal control solution'])
    plt.xlabel('$t$')
    plt.ylabel('$u(t)=dv/dt$')


    plt.figure()
    plt.plot(t_sim, L_sim, 'g-')
    plt.plot(t_sol, L_sol, 'go')
    plt.legend(['simulation result', 'optimal control solution'])
    plt.xlabel('$t$')
    plt.ylabel('$L(t)=0.5\int_{0}^{t}u^{2}dt$')

    plt.show()


