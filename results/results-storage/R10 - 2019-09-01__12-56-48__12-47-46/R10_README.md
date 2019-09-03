# BESS

--> commit: **25165d9d73351c409ef3bcd7cc7f827d24a0b38a**

###Comments:
the retailer pricing method was added and the simulations are done for both wholesale and retier model with teh same assumptions.

The retailer model is allowing for higher savings. This makes sense as the retailing price is not only much higher than the 
wholesale market one, but shows _**higher deltas**_. Hence higher savings are possible leveraging price differences with storage. 

###Assumptions:

- 14 levels all the same
- single building
- peak power = **8'000** kW
- Efficiency of the inverter = **0.95** for all techs and years
- fixed cost batteries = **1900** SGD for all techs
- retailer: Keppel electric. 
    - peak price: 20.20 cSGD/kWh, 
    - off-peak 16.16 cSGD/kWh
- IRENA: 
    - E/P ratio = 2
    - cycles per day = 1
    - electricity price = 18 USD/kWh

###To improve:

- [ ] Import also inverter efficiency
- [ ] Include the DoD. This can be made with workaround tha the storage size is actually lower.

