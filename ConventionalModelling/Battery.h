#pragma once
class lithium_ion_battery
{
public:
	//Constants, will depend on the battery-can be changed in batteries.cpp
	static double  E_0_at_ambient_temperature;
	static double Arrhenius_rate_constant_alpha; //Arrhenius rate constant for the polarization resistance.
	static double  Arrhenius_rate_constant_beta;  //Arrhenius rate constant for the internal resistance.
	static double  K; //polarisation constant
	static double  maximum_battery_capacity;
	static double  delQ_delT;
	static double  reference_temperature;  //298.15 K = 25 degC, assumed to be nominal ambient tempetature
	static double R_at_reference_temperature;
	//main funcitons
	double losses(double current, double  ambient_temperature, double internal_temperature, bool aging_effects);
	double discharge_mode();
	double charge_mode(double current, double ampere_hours_in_battery, double low_frequency_current_dynamics);
	
	//temperature effects basic equations, these are used in the temperature effect functions
	double R_of_T(double internal_temperature);
	double maximum_battery_capacity_of_T_ambient(double ambient_temperature);
	double K_of_T(double internal_temperature, double ambient_temperature);
	double E_0_of_T(double internal_temperature, double ambient_temperature);

	//temperature effects
	double charge_temperature_effects();
	double discharge_temperature_effects();
	double charge_temperature_voltage(double current, double ampere_hours_in_battery, double low_frequency_current_dynamics, double internal_temperature, double ambient_temperature);
	double discharge_temperature_voltage(double current, double ampere_hours_in_battery, double low_frequency_current_dynamics, double internal_temperature, double ambient_temperature);

	
private:
	
};


