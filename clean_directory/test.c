//
// Created by sina-mazaheri on 2/27/25.
//
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <string.h>

#define DEVICE "/dev/mychardev"
// This should match the definition in the kernel module.
#define MY_IOCTL_CMD _IOW('a', 'a', int32_t*)

int main(void) {
    int fd;
    int32_t data = 123;
    char buffer[100];
    ssize_t ret;

    // Open the device file.
    fd = open(DEVICE, O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        return errno;
    }
    printf("Device opened successfully.\n");

    // Write data to the device.
    ret = write(fd, "Hello", 5);
    if (ret < 0) {
        perror("Failed to write to device");
        close(fd);
        return errno;
    }
    printf("Wrote %zd bytes to device.\n", ret);

    // Read data from the device.
    // Note: Our module's read function is a stub and returns 0.
    ret = read(fd, buffer, sizeof(buffer) - 1);
    if (ret < 0) {
        perror("Failed to read from device");
        close(fd);
        return errno;
    }
    buffer[ret] = '\0'; // Null-terminate the string.
    printf("Read %zd bytes from device: %s\n", ret, buffer);

    // Issue an ioctl command.
    ret = ioctl(fd, MY_IOCTL_CMD, &data);
    if (ret < 0) {
        perror("Failed to issue ioctl command");
        close(fd);
        return errno;
    }
    printf("ioctl command executed successfully.\n");

    // Close the device.
    close(fd);
    printf("Device closed.\n");

    return 0;
}
