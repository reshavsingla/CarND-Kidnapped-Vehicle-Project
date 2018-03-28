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
    num_particles = 100;
    normal_distribution<double> dist_x(x, std[1]);
    normal_distribution<double> dist_y(y, std[2]);
    normal_distribution<double> dist_theta(theta, std[3]);

    weights.resize(num_particles,1.0);
    default_random_engine gen;

    for (int i = 0; i < num_particles; ++i) {
        Particle particle;
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particles.push_back(particle);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;

    for (int i = 0; i < num_particles; ++i) {
        Particle particle = particles[i];

        if(yaw_rate > 0.0001) {
            particle.x = particle.x +
                         (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
            particle.y = particle.y +
                         (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
            particle.theta = particle.theta + yaw_rate * delta_t;
        }else{
            particle.x = particle.x +
                         velocity * cos(particle.theta) * delta_t;
            particle.y = particle.y +
                         velocity * sin(particle.theta) * delta_t;
            particle.theta = particle.theta + yaw_rate * delta_t;
        }

        normal_distribution<double> dist_x(particle.x, std_pos[1]);
        normal_distribution<double> dist_y(particle.y, std_pos[2]);
        normal_distribution<double> dist_theta(particle.theta, std_pos[3]);

        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);

        particles[i] = particle;
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    vector<double> distances;
    for (int i = 0; i < predicted.size(); ++i) {
        double min = 1000000.0;
        int min_index = 0;

        if (observations.size() > 0) {
            for (int j = 0; j < observations.size(); ++j) {
                double dis = dist(predicted[i].x, predicted[i].y, observations[j].x, observations[j].y);
                if (dis < min) {
                    min = dis;
                    min_index = j;
                }
            }
            observations[min_index].id = i;
        }
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
    vector<LandmarkObs> obvs = observations;

    for (int i = 0; i < num_particles; ++i) {
        Particle particle = particles[i];

        vector<LandmarkObs> predicted;

        for (auto obs : obvs) {
            double x_c = obs.x;
            double y_c = obs.y;
            obs.x = particle.x + cos(particle.theta) * x_c - sin(particle.theta) * y_c;
            obs.y = particle.y + sin(particle.theta) * x_c + cos(particle.theta) * y_c;
            if((obs.x < particle.x + sensor_range and obs.x > particle.x - sensor_range) and (obs.y < particle.y + sensor_range and obs.y > particle.y - sensor_range)){
                predicted.push_back(obs);
            }
        }


        vector<LandmarkObs> map_observations;

        for (auto map_landmark : map_landmarks.landmark_list) {
            LandmarkObs obs{};
            obs.id = -1;
            obs.x = map_landmark.x_f;
            obs.y = map_landmark.y_f;
            if((obs.x < particle.x + sensor_range and obs.x > particle.x - sensor_range) and (obs.y < particle.y + sensor_range and obs.y > particle.y - sensor_range)) {
                map_observations.push_back(obs);
            }
        }

        dataAssociation(predicted, map_observations);

        weights[i] = 1.0;
        double std_x = std_landmark[0];
        double std_y = std_landmark[1];

        for (auto &map_observation : map_observations) {
            if(map_observation.id > -1) {
                LandmarkObs associated_prediction  = predicted[map_observation.id];
                double gauss_norm = (1 / (2 * M_PI * std_x * std_y));
                double exponent = (pow((associated_prediction.x - map_observation.x), 2) / (2 * pow(std_x, 2)) +
                                   (pow((associated_prediction.y -
                                         map_observation.y), 2)) / (2 * pow(std_y, 2)));

                double weight = gauss_norm * exp(-exponent);
                weights[i] = weights[i] * weight;
            }
        }

        particle.weight = weights[i];
        particles[i] =particle;
    }

    double total_weight = 0.0;
    for (auto& n : weights){
        total_weight += n;
    }

    cout<<total_weight<<endl;

    for (auto& n : weights) {
        n = n / total_weight;
    }


}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(weights.begin(),weights.end());
    std::vector<Particle> resampled_particles;

    for(int n=0; n<num_particles; ++n) {
        Particle particle = particles[d(gen)];
        resampled_particles.push_back(particle);
    }
    particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
