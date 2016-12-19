#ifndef PARSER_H
#define PARSER_H
#include "network.h"

network parse_network_cfg(char *filename);
void write_weights(network net, char *filename);
void write_weights_upto(network net, char *filename, int cutoff);
#endif
