#pragma once

#include "common.h"

bool file_exist(const std::string &filename) {
  int fd = shm_open(filename.c_str(), O_RDONLY, S_IRUSR | S_IWUSR);
  if (fd >= 0) {
    close(fd);
    return true;
  } else {
    return false;
  }
}