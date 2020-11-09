#include "Converter.h"

double converter::basic_circuit_and_loss() {


}

double converter::conduction_losses() {


}

double converter::switching_losses() {


}

double converter::total_losses() {
	std::vector<double> losses;

	losses.push_back(basic_circuit_and_loss());
	losses.push_back(conduction_losses());
	losses.push_back(switching_losses());

	return std::accumulate(losses.begin(), losses.end(), 0);
}