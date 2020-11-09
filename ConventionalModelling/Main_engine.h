#pragma once
#include <vector>
#include"Utilities.h"

class Spec
{
public:
	int number_of_generating_sets = 4;
	double MCR = 3840; //kW
	double maximum_continious_alternator_rating = 3685; //kW
	double alternator_efficiency = 0.97;
	double rated_output_electric = 4610; //kVA
	double engine_speed_at_MCR = 720; //rpm
	int number_of_cylinders = 8; 
	double cylinder_bore = 320; //mm
	double piston_bore = 400; //mm
	double break_mean_effective_P_at_MCR = 24.9; //bar
	double specific_fuel_oil_consumption_at_MCR = 183; //g/kWh +5%
	double specific_lub_oil_consumption_at_MCR = 0.8; // g/kWh

	
	std::vector<std::tuple<double, double>> sfc_table {(192,350),(384,347.24),(576,286.9),(768,265.9),(960,247.6),(1152,239.2),(1344,229.9),(1728,222),(1920,217.6),(2112,211.2),(2304,211.9),(2496,209.6),(2688,208.4),(2880,208.2),(3072,207.9),(3264,205),(3456,204.3),(3648,187.9),(3840,183)};
	

private:

};

class engine
{
public:
	double spesific_fuel_consumption(double power);


private:

};
