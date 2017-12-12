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
  num_particles = 75;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  default_random_engine gen;

  particles = vector<Particle>(num_particles);
  for (int i = 0; i < num_particles; i++) {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_theta(0, std_pos[2]);

  const double yaw_times_delta = yaw_rate * delta_t;

  for (Particle& particle : particles) {
    if (yaw_rate == 0) {
      particle.x += velocity * cos(particle.theta) * delta_t;
      particle.y += velocity * sin(particle.theta) * delta_t;
    } else {
      const double velocity_over_yaw = velocity / yaw_rate;
      particle.x += velocity_over_yaw
        * (sin(particle.theta + yaw_times_delta) - sin(particle.theta));
      particle.y += velocity_over_yaw 
        * (cos(particle.theta) - cos(particle.theta + yaw_times_delta));
      particle.theta += yaw_times_delta;
    }

    // Adding Gaussian noise
    particle.x += noise_x(gen);
    particle.y += noise_y(gen);
    particle.theta += noise_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.
  
  for (LandmarkObs& observation : observations) {
    double closest_distance = numeric_limits<double>::max();
    for (int i = 0; i < predicted.size(); i++) {
      const double distance = dist(predicted[i].x, predicted[i].y, observation.x, observation.y);
      if (distance < closest_distance) {
        closest_distance = distance;
        observation.id = i;
      }
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

  // for each particle, convert all observations around it into map coordinates
  // compute weights for each observations and multiply them
  // 
  // for each observation, convert into map coordinate system.
  for (Particle& particle : particles) {

    // convert from vehicle to map coordinates
    vector<LandmarkObs> obs_to_map_coords;
    for (const LandmarkObs& obs : observations) {
      double map_x = particle.x + cos(particle.theta)*obs.x - sin(particle.theta)*obs.y;
      double map_y = particle.y + sin(particle.theta)*obs.x + cos(particle.theta)*obs.y;
      obs_to_map_coords.push_back({obs.id, map_x, map_y});
    }

    // Find landmarks in range of the particle
    vector<LandmarkObs> landmarks_in_range;
    for (auto& possible_landmark : map_landmarks.landmark_list) {
      double distance = dist(particle.x, particle.y, possible_landmark.x_f, possible_landmark.y_f);
      if (distance <= sensor_range) {
        landmarks_in_range.push_back({possible_landmark.id_i, possible_landmark.x_f, possible_landmark.y_f});
      }
    }

    dataAssociation(landmarks_in_range, obs_to_map_coords);

    // Compute weights
    const double std_x = std_landmark[0];
    const double std_y = std_landmark[1];
    const double PI = 2 * acos(0.0);
    const double normalizer = 1.0 / (2 * PI * std_x * std_y);
    for (LandmarkObs& obs : obs_to_map_coords) {
      const double dist_x = particle.x - map_landmarks.landmark_list[obs.id].x_f;
      const double dist_y = particle.y - map_landmarks.landmark_list[obs.id].y_f;
      const double weight = normalizer * exp(-((dist_x*dist_x)/(2*std_x*std_x) + (dist_y*dist_y)/(2*std_y*std_y)));
      particle.weight *= weight;
    }
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
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
