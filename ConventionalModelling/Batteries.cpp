#include "Battery.h"
#include "Utilities.h"
//Constants
double lithium_ion_battery::E_0_at_ambient_temperature = 0;
double lithium_ion_battery::Arrhenius_rate_constant_alpha=0; //Arrhenius rate constant for the polarization resistance.
double lithium_ion_battery::Arrhenius_rate_constant_beta=0;  //Arrhenius rate constant for the internal resistance.
double lithium_ion_battery::K=0; //polarisation constant
double lithium_ion_battery::maximum_battery_capacity=0;
double lithium_ion_battery::delQ_delT=0;
double lithium_ion_battery::reference_temperature=298.15;
double lithium_ion_battery::R_at_reference_temperature = 0; 



double lithium_ion_battery::charge_mode(double current, double ampere_hours_in_battery, double low_frequency_current_dynamics) {




	
	return E_0 - K * (Q / (ampere_hours_in_battery + 0.1*Q))*low_frequency_current_dynamics*-K * ampere_hours_in_battery + A * exp(-B * ampere_hours_in_battery);
	
}

double lithium_ion_battery::discharge_mode(double current, double ampere_hours_in_battery, double low_frequency_current_dynamics) {

	return E_0 - K * (maximum_battery_capacity / (maximum_battery_capacity-ampere_hours_in_battery))*low_frequency_current_dynamics*-K * ampere_hours_in_battery + A * exp(-B * ampere_hours_in_battery);
}

double lithium_ion_battery::losses(double current, double ampere_hours_in_battery, double low_frequency_current_dynamics) {
	if (current > 0){
		//charge mode

	}
	else {
		//discharge mode


	}

}

double lithium_ion_battery::charge_temperature_effects() {

}

double lithium_ion_battery::charge_temperature_voltage() {
	return 
}

double lithium_ion_battery::E_0_of_T(double internal_temperature, double ambient_temperature) {
	return lithium_ion_battery::E_0_at_ambient_temperature; // + delE / delT (temperaute-ambient_temperature);

}

double lithium_ion_battery::K_of_T(double internal_temperature, double ambient_temperature) {
	return lithium_ion_battery::K_at_ambient_temperature * exp(lithium_ion_battery::Arrhenius_rate_constant_alpha*((1 / internal_temperature) - (1 / ambient_temperature));

}

double lithium_ion_battery::maximum_battery_capacity_of_T_ambient(double ambient_temperature) {

	return lithium_ion_battery::maximum_battery_capacity + lithium_ion_battery::delQ_delT*(ambient_temperature - lithium_ion_battery::reference_temperature);
}

double lithium_ion_battery::R_of_T(double internal_temperature) {

	return lithium_ion_battery::R_at_reference_temperature * exp(lithium_ion_battery::Arrhenius_rate_constant_beta*((1 / internal_temperature) - (1 / lithium_ion_battery::reference_temperature));
}

double lithium_ion_battery::charge_temperature_voltage(double current, double ampere_hours_in_battery, double low_frequency_current_dynamics, double internal_temperature, double ambient_temperature) {
	double charge_model = charge_temperature_effects();
	return charge_model - R_of_T(internal_temperature)*current;
}

double lithium_ion_battery::discharge_temperature_voltage(double current, double ampere_hours_in_battery, double low_frequency_current_dynamics, double internal_temperature, double ambient_temperature) {
	double discharge_model= discharge_temperature_effects();
	return discharge_model - R_of_T(internal_temperature)*current;

}