"""

Equations:
    Q_water = m_water * c_water * (Tw2 - Tw1)     
    Q_evap  = m_air   * c_air   * (Ta1 - Ta2)     
    COP     = eta_carnot * T_cond / (T_cond - T_evap)      (temperatures in K)
    Q_cond  = Q_evap * COP / (COP - 1)            
    P_el    = Q_cond / COP

Units: kW, kg/s, degC.
"""
from dataclasses import dataclass

C_WATER = 4.186   # kJ/(kg K)
C_AIR = 1.006     # kJ/(kg K)


@dataclass
class HeatPump:
    # ---- change these to simulate a different heat pump ----------------
    q_max: float = 5.0         # max heat output                       [kW]
    eta_carnot: float = 0.50   # share of the ideal Carnot COP reached  [-]
    pinch_evap: float = 5.0    # refrigerant is this much colder than Ta2 [K]
    pinch_cond: float = 5.0    # refrigerant is this much hotter than Tw2 [K]

    def run(self, Ta1, Ta2, Tw1, Tw2, m_air, m_water, power_from_PV=0.0):
        """
        Ta1, Ta2 : air temperature in / out of the evaporator   [degC]
        Tw1, Tw2 : water temperature in / out of the condenser  [degC]
        m_air    : air mass flow                                [kg/s]
        m_water  : water mass flow                              [kg/s]
        power_from_PV : electric power available from PV        [kW]
        """
        # 1) heat the water loop takes
        Q_water = m_water * C_WATER * (Tw2 - Tw1)

        # 2) COP from the refrigerant temperatures [K]
        T_evap = Ta2 - self.pinch_evap + 273.15
        T_cond = Tw2 + self.pinch_cond + 273.15
        cop = self.eta_carnot * T_cond / (T_cond - T_evap)

        # 3) most heat the air side can supply as Q_cond
        Q_evap_air = m_air * C_AIR * (Ta1 - Ta2)
        Q_air_limit = Q_evap_air * cop / (cop - 1)

        # 4) result: smallest limit wins
        limits = {"water": Q_water, "air": Q_air_limit, "size": self.q_max}
        limiting_factor = min(limits, key=limits.get)
        Q_cond = limits[limiting_factor]

        # 5) electricity and heat from air at this Q_cond
        P_el = Q_cond / cop
        Q_evap = Q_cond - P_el
        power_from_grid = max(P_el - power_from_PV, 0.0)

        return {
            "Q_cond": Q_cond,                 # heat delivered to water   [kW]
            "limiting_factor": limiting_factor,
            "COP": cop,
            "P_el": P_el,                     # electric power            [kW]
            "Q_evap": Q_evap,                 # heat taken from air       [kW]
            "power_from_grid": power_from_grid,
        }


if __name__ == "__main__":
    hp = HeatPump()

    # normal case: the water loop is what limits
    r = hp.run(Ta1=5, Ta2=2, Tw1=30, Tw2=35, m_air=1.0, m_water=0.19,
               power_from_PV=0.8)
    print(f"Q_cond = {r['Q_cond']:.2f} kW (limited by {r['limiting_factor']}), "
          f"COP = {r['COP']:.2f}, P_el = {r['P_el']:.2f} kW")

    # too little airflow: the air side limits
    r = hp.run(Ta1=5, Ta2=2, Tw1=30, Tw2=35, m_air=0.5, m_water=0.19)
    print(f"Q_cond = {r['Q_cond']:.2f} kW (limited by {r['limiting_factor']}), "
          f"COP = {r['COP']:.2f}, P_el = {r['P_el']:.2f} kW")
