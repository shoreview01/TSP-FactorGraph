# Manuscript operational EV energy model

The route oracle implements the edge model in `docs/manuscript.pdf`, Eqs. (2)-(8),
using hourly SUMO/TOPIS travel time, DEM road grade, and the current waste
payload. The implementation is in `simul/energy.py`.

For total mass `m = curb mass + payload`, mean edge speed `v = L/T`, and road
angle `theta = atan(grade)`, the wheel-energy terms are

```
E_roll  = m g c_rr cos(theta) L
E_grade = m g sin(theta) L
E_aero  = 0.5 rho Cd Af v^2 L
```

Positive wheel work is divided by propulsion efficiency.  Negative work is
credited using recuperation efficiency.  A constant auxiliary load is
integrated over travel time, so congestion incurs an idle/HVAC/hydraulic cost.

The stop-and-go term follows Eq. (8) directly:

```
T_ref = L / v_free
Delta_T = 0.25 T_ref
phi = max((T - T_ref) / Delta_T - 1, 0)
E_stop-go = kappa_sg (m0 + payload) phi
```

Thus, no extra energy is added until edge traversal time exceeds 1.25 times
the free-flow reference. The coefficient
`kappa_sg=4.93e-6 kWh/kg` is the unrecovered battery energy per kilogram
for one 30 km/h acceleration-braking cycle under the Table-I efficiencies
`eta_drv=0.90` and `eta_reg=0.60`.

At a serviced bin the battery additionally supplies

```
E_service = E_lift/compact + P_service * service_time.
```

Default truck parameters are: 7,000 kg curb mass, 2,000 kg payload capacity,
`c_rr=0.01`, `Cd=0.60`, `Af=5.5 m^2`, `rho=1.225 kg/m^3`,
propulsion efficiency 0.90, recuperation efficiency 0.60, 3 kW travel
auxiliary load, 2 kW service auxiliary load, 120 s service time, and
0.12 kWh lifting/compaction energy per bin.

For shortest-path optimization each edge cost is kept nonnegative.  If a steep
downhill edge would regenerate more than its traction and auxiliary use, the
excess is recorded as curtailed recuperation instead of creating a negative
edge.  This is conservative and avoids nonphysical energy-generating cycles in
the routing graph.

References:

- https://sumo.dlr.de/docs/Models/Electric.html
- https://github.com/eclipse-sumo/sumo/blob/main/src/utils/emissions/HelpersEnergy.cpp
