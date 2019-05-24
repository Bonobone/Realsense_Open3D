#include <GL/glew.h>
#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <librealsense2/rs.hpp>
#include <algorithm>
#include "example.hpp"
#include <fstream>

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <Open3D/Open3D.h>

#include <time.h>


//Helper class for controlling the filter's GUI element
struct filter_slider_ui{
	std::string name;
	std::string label;
	std::string description;
	bool is_int;
	float value;
	rs2::option_range range;
};

//Class to encapsulate a filter alongside its options
class filter_options {
public:
	filter_options(const std::string name, rs2::filter& filter);
	filter_options(filter_options&& other);
	std::string filter_name;
	rs2::filter& filter;
	std::map<rs2::options, filter_slider_ui> supported_options;
	std::atomic_bool is_enabled;
};

//Helper function for getting data from the queues and updating the view
void update_data(rs2::frame_queue& data, rs2::frame& depth, rs2::points& points, rs2::pointcloud& pc, glfw_state& view, rs2::frame_queue& color_queue, bool* saveflug);

//Helper function for get RGB values based on normals - tex coords, normal value [u, v]
std::tuple<uint8_t, uint8_t, uint8_t> get_texcolor(rs2::video_frame texture, rs2::texture_coordinate texcoords);

int main(int argc, char* argv[]) try {

	//Create a simple OpenGL Window
	window app(1280, 720, "RealSense Post Processing and Save PCD Example");

	//Construct objects to manage view state
	glfw_state original_view_orientation{};
	glfw_state filtered_view_orientation{};
	filtered_view_orientation.yaw = 0.0f;
	filtered_view_orientation.pitch = 0.0f;

	//Declear pointcloud objects, for calclating pontclouds texture mappings
	rs2::pointcloud original_pc;
	rs2::pointcloud filtered_pc;

	//Declear RealSense pipeline, encupslating hte actual device and sensors
	rs2::pipeline pipe;
	rs2::config cfg;

	//Use a configulation object to result
	cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 0, RS2_FORMAT_Z16, 30);
	cfg.enable_stream(RS2_STREAM_COLOR, 1280, 0, RS2_FORMAT_RGB8, 30);

	//Start streaming with above configulation
	pipe.start(cfg);

	//Declarelation of filters
	rs2::decimation_filter dec_filter;
	rs2::threshold_filter thr_filter;
	rs2::spatial_filter spat_filter;
	rs2::temporal_filter temp_filter;

	const std::string disparity_filter_name = "Disparity";
	rs2::disparity_transform depth_to_disparity(true);
	rs2::disparity_transform disparity_to_depth(false);

	//Initiate a vector that holds filters and their options
	std::vector<filter_options> filters;

	//The following order of emplacement will dictate the orders in which filters are applied
	filters.emplace_back("Decimate", dec_filter);
	filters.emplace_back("Threshold", thr_filter);
	filters.emplace_back(disparity_filter_name, depth_to_disparity);
	filters.emplace_back("Spatial", spat_filter);
	filters.emplace_back("Temporal", temp_filter);

	//Declearing two concurrent queues that will be used to enqueue frames from different threads
	rs2::frame_queue original_data;
	rs2::frame_queue filtered_data;

	rs2::frame_queue color_data;

	//Declear depth colorizer for pretty visualization of depth data
	rs2::colorizer color_map;

	//Atomic bool to allow thread safe way to stop the thread
	std::atomic_bool stopped(false);

	bool saveflug = 0;
	auto psaveflug = &saveflug;

	//Create a thread for getting frames from the device and process them
	//to prevent UI thread from blocking due to long computations
	std::thread processing_thread([&]() {
		while (!stopped) {
			rs2::frameset data = pipe.wait_for_frames();
			rs2::frame depth_frame = data.get_depth_frame();
			if (!depth_frame)
				return;
			rs2::frame color_frame = data.get_color_frame();
			if (!color_frame)
				return;

			rs2::frame filtered = depth_frame;	//Does not copy the frame, only adds a reference

			//Apply filters

			//threshold filter - Set thresold of distance
			thr_filter.set_option(RS2_OPTION_MIN_DISTANCE, 0.1f);
			thr_filter.set_option(RS2_OPTION_MAX_DISTANCE, 8.0f);

			//temporal filter - Filters depth data by looking previous frames
			temp_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.06f);
			temp_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20.0f);

			//decimation filter - Reduce the resolution of depth frame
			dec_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 1.0f);

			//spatial filter - Edge-preserving smoothing of of depth data
			spat_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2.0f);
			spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.34f);
			spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 10.0f);

			//disparity - Performs transformation between depth and disparity domains
			bool revert_disparity = false;
			for (auto&& filter : filters) {
				filtered = filter.filter.process(filtered);
				if (filter.filter_name == disparity_filter_name) {
					revert_disparity = true;
				}
			}
			if (revert_disparity) {
				filtered = disparity_to_depth.process(filtered);
			}

			//Push filtered and original data to their respective queues
			filtered_data.enqueue(filtered);
			color_data.enqueue(color_frame);

		}

		}
		
	);
	
	//Declear objects that will hold the calclated pointclouds and colored frames
	rs2::frame colored_depth;
	rs2::frame colored_filtered;
	rs2::points original_points;
	rs2::points filtered_points;

	while (app) {
		float w = static_cast<float>(app.width());
		float h = static_cast<float>(app.height());

		glEnable(GLFW_STICKY_KEYS);

		if (glfwGetKey(app, 'W')) {
			filtered_view_orientation.pitch += 0.1;
		}
		if (glfwGetKey(app, 'A')) {
			filtered_view_orientation.yaw += 0.1;
		}
		if (glfwGetKey(app, 'S')) {
			filtered_view_orientation.pitch -= 0.1;
		}
		if (glfwGetKey(app, 'D')) {
			filtered_view_orientation.yaw -= 0.1;
		}
		if (glfwGetKey(app, GLFW_KEY_SPACE)) {
			filtered_view_orientation.offset_y += 0.1;
		}
		if (glfwGetKey(app, GLFW_KEY_LEFT_SHIFT)) {
			filtered_view_orientation.offset_y -= 0.1;
		}
		if (glfwGetKey(app, 'I')) {
			filtered_view_orientation.up_down += 0.1;
		}
		if (glfwGetKey(app, 'J')) {
			filtered_view_orientation.left_right += 0.1;
		}
		if (glfwGetKey(app, 'K')) {
			filtered_view_orientation.up_down -= 0.1;
		}
		if (glfwGetKey(app, 'L')) {
			filtered_view_orientation.left_right -= 0.1;
		}

		if (glfwGetKey(app, GLFW_KEY_LEFT_CONTROL)) {	//Ctrl + S
			glDisable(GLFW_STICKY_KEYS);
			if (glfwGetKey(app, 'S')) {
				saveflug = 1;
			}
		}

		if (glfwGetKey(app, GLFW_KEY_ESCAPE)) {		//ESC
			app.close();
		}

		update_data(filtered_data, colored_filtered, filtered_points, filtered_pc, filtered_view_orientation, color_data, psaveflug);

		if (filtered_points) {
			draw_pointcloud(int(w), int(h), filtered_view_orientation, filtered_points);
		}

	}

	//Signal the processing thread to stop, and join
	stopped = true;
	processing_thread.join();
	pipe.stop();

	return EXIT_SUCCESS;

}
catch (const rs2::error& e) {
	std::cerr << "RealSense Sensor Calling : " << e.get_failed_function() << "(" << e.get_failed_args() << ") : \n" << e.what() << std::endl;
	return EXIT_FAILURE;
}
catch (const std::exception& e) {
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}

filter_options::filter_options(const std::string name, rs2::filter& filter) :
	filter_name(name),
	filter(filter),
	is_enabled(true)
{
	const std::array<rs2_option, 5> possible_filter_options = {
		RS2_OPTION_FILTER_MAGNITUDE,
		RS2_OPTION_FILTER_SMOOTH_ALPHA,
		RS2_OPTION_MIN_DISTANCE,
		RS2_OPTION_MAX_DISTANCE,
		RS2_OPTION_FILTER_SMOOTH_DELTA
	};
}

filter_options::filter_options(filter_options&& other) :
	filter_name(std::move(other.filter_name)),
	filter(other.filter),
	supported_options(std::move(other.supported_options)),
	is_enabled(other.is_enabled.load())
{
}

void update_data(rs2::frame_queue& data, rs2::frame& colorized_depth, rs2::points& points, rs2::pointcloud& pc, glfw_state& view, rs2::frame_queue& color_queue, bool* psaveflug) {
	rs2::frame f;
	rs2::frame c;

	if (data.poll_for_frame(&f)) {				//Try to take the depth and points from the queue
		if (color_queue.poll_for_frame(&c)) {	//Try to the color frame from the queue
			pc.map_to(c);						//Map the colored depth to the pointcloud
			points = pc.calculate(f);			//Generate pointcloud from the depth data
			view.tex.upload(c);					//Upload the texture to the view (without this the view will be Black & White)


			//Save pointcloud if(saveflug)
			if (*psaveflug == 1) {
				open3d::geometry::PointCloud OPC;
				auto vertices = points.get_vertices();
				auto tex_coordinates = points.get_texture_coordinates();
				
				for (int d = 0; d < points.size(); ++d) {
					if (vertices[d].z != 0.0) {
						OPC.points_.emplace_back(Eigen::Vector3d(vertices[d].x, -1.0 * vertices[d].y, -1.0 * vertices[d].z));	//Add the point's coordinate

						std::tuple<uint8_t, uint8_t, uint8_t> current_color;	//tuple for 8-bit color
						current_color = get_texcolor(c, tex_coordinates[d]);	//Get color of this point from color frame and texture coordinates [u, v]
						OPC.colors_.emplace_back(Eigen::Vector3d((double)std::get<0>(current_color) / 255.0, (double)std::get<1>(current_color) / 255.0, (double)std::get<2>(current_color) / 255.0 ));	//Add the point's color
							//Note: The RGB data of open3D::Pointcloud.colors_ is normarized from 0 to 1. (double)
							//		So 8-bit color data is  needed to cast to double and devided by 255.0 to normarize.
						//open3d::geometry::CreatePointCloudFromDepthImage();
					}
				}

				auto now = std::time(nullptr);
				struct tm* tmNow = std::localtime(&now);
				
				std::string filename = "PCD_" + std::to_string( (1900 + tmNow->tm_year) ) + std::to_string(tmNow->tm_mon) + std::to_string(tmNow->tm_mday) + "_" 
										+ std::to_string(tmNow->tm_hour) + ":" + std::to_string(tmNow->tm_min) + ":" + std::to_string(tmNow->tm_sec) + ".pcd";

				if (open3d::io::WritePointCloudToPCD(filename, OPC, true, false)) {
					std::cout << "Save Suceeded" << std::endl;

				}
				else {
					std::cerr << "Save failed" << std::endl;
				}

				*psaveflug = 0;

			}
		}
	}

}

//Helper function for get RGB values based on normals - tex coords, normal value [u, v]
std::tuple<uint8_t, uint8_t, uint8_t> get_texcolor(rs2::video_frame texture, rs2::texture_coordinate texcoords) {
	const int w = texture.get_width(), h = texture.get_height();

	//Convert normals [u, v] to basic coordinates [x, y]
	int x = std::min(std::max(int(texcoords.u * w + .5f), 0), w - 1);
	int y = std::min(std::max(int(texcoords.v * h + .5f), 0), h - 1);

	//Calclate index of color frame
	int idx = x * texture.get_bytes_per_pixel() + y * texture.get_stride_in_bytes();

	//Get pointer of color frame
	const auto texture_data = reinterpret_cast<const uint8_t*>(texture.get_data());

	return std::tuple<uint8_t, uint8_t, uint8_t>(texture_data[idx], texture_data[idx + 1], texture_data[idx + 2]);
}