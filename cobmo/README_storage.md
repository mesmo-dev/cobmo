# README storage
Here are reported the main comments and notes about the several storage algorithms.

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
            self.tank_geometry(
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