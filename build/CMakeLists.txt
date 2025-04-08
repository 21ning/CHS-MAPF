cmake_minimum_required(VERSION 3.0)
project(dliteplanner)

# 找到pybind11
find_package(pybind11 REQUIRED)

pybind11_add_module(dliteplanner dlite_planner_py.cpp)
