#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "activation_layer.h"
#include "activations.h"
#include "assert.h"
#include "avgpool_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "crnn_layer.h"
#include "crop_layer.h"
#include "detection_layer.h"
#include "dropout_layer.h"
#include "gru_layer.h"
#include "list.h"
#include "local_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "reorg_layer.h"
#include "rnn_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "softmax_layer.h"
#include "utils.h"

typedef struct{
    char *type;
    list *options;
}section;

list *read_cfg(char *filename);




typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network net;
} size_params;


/*
 * added by huangguoxiong
 */
void write_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //save_convolutional_weights_binary(l, fp);
        //return;
    }
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    int num = l.n*l.c*l.size*l.size;
    
	int j;
	fprintf(fp,"biases,");
	for(j = 0; j < l.n; j++){
		//fwrite(l.biases, sizeof(float), l.n, fp);
		fprintf(fp,"%f,",l.biases[j]);
	}
	fprintf(fp,"\n");
    if (l.batch_normalize){
		fprintf(fp,"scales,");
		for(j = 0; j < l.n; j++){
			//fwrite(l.scales, sizeof(float), l.n, fp);
			fprintf(fp,"%f,",l.scales[j]);
		}
		fprintf(fp,"\n");
		fprintf(fp,"rolling_mean,");
		for(j = 0; j < l.n; j++){			
			//fwrite(l.rolling_mean, sizeof(float), l.n, fp);
			fprintf(fp,"%f,",l.rolling_mean[j]);
		}
		fprintf(fp,"\n");
		fprintf(fp,"rolling_variance,");
		for(j = 0; j < l.n; j++){				
			//fwrite(l.rolling_variance, sizeof(float), l.n, fp);
			fprintf(fp,"%f,",l.rolling_variance[j]);
		}
		fprintf(fp,"\n");
    }
    //fwrite(l.weights, sizeof(float), num, fp);
	fprintf(fp,"weights,");
	for(j = 0; j < num; j++){
		fprintf(fp,"%f,",l.weights[j]);
	}
	fprintf(fp,"\n");
    if(l.adam){
		fprintf(fp,"m,");
		for(j = 0; j < num; j++){
			//fwrite(l.m, sizeof(float), num, fp);
			fprintf(fp,"%f,",l.m[j]);
		}
		fprintf(fp,"\n");
		fprintf(fp,"v,");
		for(j = 0; j < num; j++){				
			//fwrite(l.v, sizeof(float), num, fp);
			fprintf(fp,"%f,",l.v[j]);
		}
		fprintf(fp,"\n");
    }
	
	
}





/*
 * added by huangguoxiong
 */
FILE * get_file_from_name(char *filename)
{
	FILE *fp = fopen(filename, "w");
    if(!fp) file_error(filename);
	return fp;
}

/*
 * added by huangguoxiong
 */
void write_weights_upto(network net, char *filename, int cutoff)
{
#ifdef GPU
    if(net.gpu_index >= 0){
        cuda_set_device(net.gpu_index);
    }
#endif
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "w");
    if(!fp) file_error(filename);

    int major = 0;
    int minor = 1;
    int revision = 0;
    //fwrite(&major, sizeof(int), 1, fp);
	fprintf(fp,"major=%d,",major);
    //fwrite(&minor, sizeof(int), 1, fp);
	fprintf(fp,"minor=%d,",minor);
    //fwrite(&revision, sizeof(int), 1, fp);
	fprintf(fp,"revision=%d,",revision);
    //fwrite(net.seen, sizeof(int), 1, fp);
	fprintf(fp,"net.seen=%d,",net.seen[0]);

    int i;
	int j;
	char fn[50];
	       
    for(i = 0; i < net.n && i < cutoff; ++i){
		sprintf(fn, "hgx/%d_conv_normalize.txt", i);
		
		FILE *split_fp = stderr;
	
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
			sprintf(fn, "hgx/%d_conv.txt", i);
			split_fp = get_file_from_name(fn);
            write_convolutional_weights(l, split_fp);
        } if(l.type == CONNECTED){
			sprintf(fn, "hgx/%d_connected.txt", i);
			split_fp = get_file_from_name(fn);
            save_connected_weights(l, split_fp);
        } if(l.type == BATCHNORM){
			sprintf(fn, "hgx/%d_batchnorm.txt", i);
			split_fp = get_file_from_name(fn);			
            save_batchnorm_weights(l, split_fp);
        } if(l.type == RNN){
			sprintf(fn, "hgx/%d_RNN.txt", i);
			split_fp = get_file_from_name(fn);			
            save_connected_weights(*(l.input_layer), split_fp);
            save_connected_weights(*(l.self_layer), split_fp);
            save_connected_weights(*(l.output_layer), split_fp);
        } if(l.type == GRU){
			sprintf(fn, "hgx/%d_GRU.txt", i);
			split_fp = get_file_from_name(fn);				
            save_connected_weights(*(l.input_z_layer), split_fp);
            save_connected_weights(*(l.input_r_layer), split_fp);
            save_connected_weights(*(l.input_h_layer), split_fp);
            save_connected_weights(*(l.state_z_layer), split_fp);
            save_connected_weights(*(l.state_r_layer), split_fp);
            save_connected_weights(*(l.state_h_layer), split_fp);
        } if(l.type == CRNN){
			sprintf(fn, "hgx/%d_CRNN.txt", i);
			split_fp = get_file_from_name(fn);				
            save_convolutional_weights(*(l.input_layer), split_fp);
            save_convolutional_weights(*(l.self_layer), split_fp);
            save_convolutional_weights(*(l.output_layer), split_fp);
        } if(l.type == LOCAL){
#ifdef GPU
            if(gpu_index >= 0){
                pull_local_layer(l);
            }
#endif
			sprintf(fn, "hgx/%d_other.txt", i);
			split_fp = get_file_from_name(fn);	
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
			
            //fwrite(l.biases, sizeof(float), l.outputs, fp);
			for(j = 0; j < l.outputs; j++){
				fprintf(split_fp,"biases=%f,",l.biases[j]);
			}
            //fwrite(l.weights, sizeof(float), size, fp);
			for(j = 0; j < size; j++){
				fprintf(split_fp,"weights=%f,",l.weights[j]);
			}
		
		}
		fprintf(stderr, "%d Done!\n",i);
		fflush(split_fp);
	}
	fclose(fp);
}

/*
 * added by huangguoxiong
 */
 void write_weights(network net, char *filename)
{
    write_weights_upto(net, filename, net.n);
}
