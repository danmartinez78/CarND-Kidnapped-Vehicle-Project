/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    static default_random_engine gen;
    gen.seed(889);

    // number of particles
    num_particles = 100;

    // zero mean noise
    normal_distribution<double> x_dist(0, std[0]);
    normal_distribution<double> y_dist(0, std[1]);
    normal_distribution<double> theta_dist(0, std[2]);

    // initialize particles
    for (int i=0;i<num_particles;i++){
        Particle p;
        p.id = i;
        p.x = x + x_dist(gen);
        p.y = y+ y_dist(gen);
        p.theta = theta + theta_dist(gen);
        p.weight = 1.0/num_particles;
        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    static default_random_engine gen;
    gen.seed(456);
    // zero mean sensor noise
    normal_distribution<double> x_dist(0, std_pos[0]);
    normal_distribution<double> y_dist(0, std_pos[1]);
    normal_distribution<double> theta_dist(0, std_pos[2]);




    for(int i=0;i<num_particles;i++){
        double dx, dy, new_theta = 0.0;
        if (fabs(yaw_rate) < 0.0001) {
            dx = velocity * delta_t * cos(particles[i].theta);
            dy = velocity * delta_t * cos(particles[i].theta);
            new_theta = particles[i].theta;
        }
        else{
            new_theta = particles[i].theta + yaw_rate*delta_t;
            dx = velocity/yaw_rate * (sin(new_theta) - sin(particles[i].theta));
            dy = velocity/yaw_rate * (cos(particles[i].theta) - cos(new_theta));
        }
        particles[i].x += dx + x_dist(gen);
        particles[i].y += dy + y_dist(gen);
        particles[i].theta = new_theta + theta_dist(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

    for(int i = 0; i < observations.size(); i++){
        LandmarkObs o = observations[i];

        double min_dist = numeric_limits<double>::max();
        int best_id = -1;
        for(int j = 0; j< predicted.size(); j++){
            LandmarkObs p = predicted[j];
            double distance = dist(o.x, o.y, p.x, p.y);
            if(distance < min_dist){
                min_dist = distance;
                best_id = p.id;
            }
        }
        observations[i].id = best_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    for (int i = 0; i < num_particles; i++) {
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;

        std::vector<LandmarkObs> predicted;

        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            Map::single_landmark_s l = map_landmarks.landmark_list[j];

            if (fabs(l.x_f - p_x) <= sensor_range && fabs(l.y_f - p_y) <= sensor_range) {
                LandmarkObs l_obs = {l.id_i, l.x_f, l.y_f};
                predicted.push_back(l_obs);
            }
        }

        std::vector<LandmarkObs> transformed_observations;
        for (int j = 0; j < observations.size(); j++) {
            double x = p_x + cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y;
            double y = p_y + sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y;

            LandmarkObs l_obs = {observations[j].id, x, y};
            transformed_observations.push_back(l_obs);
        }

        dataAssociation(predicted, transformed_observations);

        double sig_x = std_landmark[0];
        double sig_y = std_landmark[1];

        particles[i].weight = 1.0;

        particles[i].associations.clear();
        particles[i].sense_x.clear();
        particles[i].sense_y.clear();

        for (int j = 0; j < transformed_observations.size(); j++) {

            double mu_x, mu_y;
            for (int k = 0; k < predicted.size(); k++) {
                if (predicted[k].id == transformed_observations[j].id) {
                    mu_x = predicted[k].x;
                    mu_y = predicted[k].y;
                    break;
                }
            }
            double x = transformed_observations[j].x;
            double y = transformed_observations[j].y;

            double gauss_norm = 1/(2*M_PI*sig_x*sig_y);
            double exponent_1 = pow(x-mu_x,2)/(2*pow(sig_x,2));
            double exponent_2 = pow(y-mu_y,2)/(2*pow(sig_y,2));

            particles[i].weight *= (gauss_norm * exp(-(exponent_1+exponent_2)));
            weights[i] = particles[i].weight;

            particles[i].associations.push_back(transformed_observations[j].id);
            particles[i].sense_x.push_back(transformed_observations[j].x);
            particles[i].sense_y.push_back(transformed_observations[j].y);
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;

    // Take a discrete distribution with pmf equal to weights
    discrete_distribution<> weights_pmf(weights.begin(), weights.end());
    // initialise new particle array
    vector<Particle> new_particles;
    for (int i = 0; i < num_particles; ++i)
        new_particles.push_back(particles[weights_pmf(gen)]);

    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

