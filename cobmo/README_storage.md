# README storage
Here are reported the main comments and notes about the several storage algorithms.

## To be improved
date: 2019-09-05
- BESS: the BESS simulations do not take into account by now the minimum state of charge given by the depth of discharge 
constraint. This is important to be included as batteries cannot fully discharge.
- TESS: the sensible storage case is not running with the retailer price signal. This has to be debugged.

## Sensible Storage
### Losses

The following piece of code is a way to express the losses of the storage due to:

- Heat loss towards the external environment. Usually negligible, depends on the insulation material of the tank and 
convection transfer coefficients. Values from the literature can be taken for `heat_transfer_coefficient` 

 

    (
        (
            4 * self.parse_parameter(self.building_scenarios['heat_transfer_coefficient'])
        )
        /
        (
            self.calculate_tank_diameter_height(
                self.parse_parameter(self.building_scenarios['storage_size']),
                self.parse_parameter(self.building_scenarios['tank_aspect_ratio'])
            )
            * self.parse_parameter('tank_fluid_density')
        )
    ) * (
        self.parse_parameter(self.building_scenarios['storage_cooling_ambient_temperature'])
        - self.parse_parameter(self.building_scenarios['storage_cooling_temperature_bottom_layer'])
    ) / (
        self.parse_parameter(self.building_scenarios['storage_sensible_total_delta_temperature_layers'])
        * self.parse_parameter('water_specific_heat')
    )
        
One might also include another term:

- Heat loss between layers. This is conduction throughout the thermocline. usually negligible. values of the UA 
(overall heat transfer coefficient) can be found in the literature

#### Other losses expression

This is another expression for losses:
        
        self.state_matrix.at[
                self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_state_of_charge',
                self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_mass_factor',
            ] = (
                self.state_matrix.at[
                    self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_state_of_charge',
                    self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_mass_factor',
                ]
            ) - (
                (
                    self.parse_parameter(self.building_scenarios['storage_UA_external'])
                    * (
                        self.parse_parameter(self.building_scenarios['storage_cooling_ambient_temperature'])
                        - self.parse_parameter(self.building_scenarios['storage_cooling_temperature_bottom_layer'])
                    )
                    / self.parse_parameter('water_specific_heat')
                    / self.parse_parameter(self.building_scenarios['storage_sensible_total_delta_temperature_layers'])
                )
                + (
                    self.parse_parameter(self.building_scenarios['storage_UA_thermocline'])
                    / self.parse_parameter('water_specific_heat')
                )
            )

## Addition of noise to state_matrix

It has been noticed that leaving to 0.0 the losses of the storage creates problems in inverting the state matrix, 
as it becomes a single matrix.

A solution tried was to add some noise to the *whole* state_matrix. This solution though was compromising the quality of the results.

It has then been opted for the addition of losses specifically in teh cell of the state_matrix 
[state_of_charge, state_of_charge], which is some kind of noise by itself.

See the below piece for the noise. 

Source: https://stackoverflow.com/questions/44305456/why-am-i-getting-linalgerror-singular-matrix-from-grangercausalitytests?rq=1

    # adding some noise to make the matrix invertible.
    # source:
    # https://stackoverflow.com/questions/44305456/why-am-i-getting-linalgerror-singular-matrix-from-grangercausalitytests?rq=1
    self.state_matrix = self.state_matrix + 1e-20 * np.random.rand(  # np.clip(
        self.state_matrix.values.shape[0], self.state_matrix.values.shape[1]
    )  # , 0e-25, 2.0)
    # The clipping command is needed to limit the random numbers between 0 and 2.0
    # source: 