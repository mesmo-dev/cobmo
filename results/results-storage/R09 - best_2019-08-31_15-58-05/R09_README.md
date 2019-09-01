# BESS

####CAREFUL! these results are run on a version with a mistake. MISTAKE: the savings calculated for the storage were based on a constant number for the baseline scenario OPEX. This should instead also change depending on the storage lifetime.
__________________________________________________________________________________
--> commit **cf244fba246ef1708beeef649c59213f9a80161d**

Assumptions:

- peak power = 8'000 kW
- Efficiency of the inverter = 0.95 for all techs and years
- fixed cost batteries = 1200 SGD for all techs

To improve:

- [ ] Import also inverter efficiency
- [x] increase fixed cost as it it was in eur 
- [ ] Include the DoD. This can be made with workaround tha the storage size is actually lower.

