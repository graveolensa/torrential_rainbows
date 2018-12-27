#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "interactions.h"
#include <iostream>
#include <chrono>
#include <thread>



#include <stdio.h>
#include <math.h>
#include "portaudio.h"

#define NUM_SECONDS   (5)
#define SAMPLE_RATE   (44100)
#define FRAMES_PER_BUFFER  (64)


#ifndef M_PI
#define M_PI  (3.14159265)
#endif

#define TABLE_SIZE   (200)


// texture and pixel objects
GLuint pbo = 0; // OpenGL pixel buffer object
GLuint tex = 0; // OpenGL texture object

struct cudaGraphicsResource *cuda_pbo_resource;

PaStream *stream;
PaError err;

double *audio_buffer;

const int BUFFER_SIZE = FRAMES_PER_BUFFER*sizeof(double)*2;

typedef struct
{

    float sine[TABLE_SIZE];
    int left_phase;
    int right_phase;
    char message[20];
}
paTestData;

void render() {
  uchar4 *d_out = 0;
  cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
  
  cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,
                                       cuda_pbo_resource);
  kernelLauncher(d_out, W, H, loc);
  cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void drawTexture() {
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
  glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H);
  glTexCoord2f(1.0f, 1.0f); glVertex2f(W, H);
  glTexCoord2f(1.0f, 0.0f); glVertex2f(W, 0);
  glEnd();
  glDisable(GL_TEXTURE_2D);
}

void display() {
  render();
  drawTexture();
  glutSwapBuffers();
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(30ms);
  loc.x++;
  loc.y++;

  glutPostRedisplay();
}

void initGLUT(int *argc, char **argv) {
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(W, H);
  glutCreateWindow(TITLE_STRING);
#ifndef __APPLE__
  glewInit();
#endif
}

void initPixelBuffer() {

  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * W*H*sizeof(GLubyte), 0,
               GL_STREAM_DRAW);
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
  cudaGraphicsMapFlagsWriteDiscard);
  audio_buffer = (double*) malloc(BUFFER_SIZE);

}

void exitfunc() {
  PaError err;
  if (pbo) {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    err = Pa_StopStream( stream );
    if( err != paNoError ) goto error;

    err = Pa_CloseStream( stream );
    if( err != paNoError ) goto error;

    Pa_Terminate();

  }
  free(audio_buffer);
error:
    Pa_Terminate();
    fprintf( stderr, "An error occured while using the portaudio stream\n" );
    fprintf( stderr, "Error number: %d\n", err );
    fprintf( stderr, "Error message: %s\n", Pa_GetErrorText( err ) );

}




static int audioCallback( const void *inputBuffer, void *outputBuffer,
                            unsigned long framesPerBuffer,
                            const PaStreamCallbackTimeInfo* timeInfo,
                            PaStreamCallbackFlags statusFlags,
                            void *userData )
{ 

  float *out = (float*)outputBuffer;
  unsigned long i;
  double d_audio_buffer;


  audioKernelLauncher(audio_buffer, FRAMES_PER_BUFFER, W, H, loc);

  
  for( i=0; i<framesPerBuffer; i++ ){
    //printf("audio_buffer[i]:%f\n", audio_buffer[i]);
    *out++ = audio_buffer[i]; /* left */
    *out++ = audio_buffer[i]; /* right */
  }

    
    return paContinue;
}



static void StreamFinished( void* userData )
{
   paTestData *data = (paTestData *) userData;
   printf( "Stream Completed: %s\n", data->message );
}



int main(int argc, char** argv) {


  PaStreamParameters outputParameters;
  paTestData data;

  printInstructions();
  initGLUT(&argc, argv);
  gluOrtho2D(0, W, H, 0);
  glutKeyboardFunc(keyboard);
  glutSpecialFunc(handleSpecialKeypress);
  glutPassiveMotionFunc(mouseMove);
  glutMotionFunc(mouseDrag);
  glutMouseFunc(mouseUpDown);
  glutDisplayFunc(display);
  initPixelBuffer();





  int i;

  /* initialise sinusoidal wavetable */
  for( i=0; i<TABLE_SIZE; i++ )
  {
      data.sine[i] = (float) sin( ((double)i/(double)TABLE_SIZE) * M_PI * 2. );
  }
  data.left_phase = data.right_phase = 0;
  
  err = Pa_Initialize();
  if( err != paNoError ) goto error;

  outputParameters.device = Pa_GetDefaultOutputDevice(); /* default output device */
  if (outputParameters.device == paNoDevice) {
    fprintf(stderr,"Error: No default output device.\n");
    goto error;
  }
  outputParameters.channelCount = 2;       /* stereo output */
  outputParameters.sampleFormat = paFloat32; /* 32 bit floating point output */
  outputParameters.suggestedLatency = Pa_GetDeviceInfo( outputParameters.device )->defaultLowOutputLatency;
  outputParameters.hostApiSpecificStreamInfo = NULL;

  err = Pa_OpenStream(
            &stream,
            NULL, /* no input */
            &outputParameters,
            SAMPLE_RATE,
            FRAMES_PER_BUFFER,
            paClipOff,      /* we won't output out of range samples so don't bother clipping them */
            audioCallback,
            &data );
  if( err != paNoError ) goto error;

  sprintf( data.message, "No Message" );
  err = Pa_SetStreamFinishedCallback( stream, &StreamFinished );
  if( err != paNoError ) goto error;

  err = Pa_StartStream( stream );
  if( err != paNoError ) goto error;


  glutMainLoop();
  atexit(exitfunc);



  
  return 0;







error:
    Pa_Terminate();
    fprintf( stderr, "An error occured while using the portaudio stream\n" );
    fprintf( stderr, "Error number: %d\n", err );
    fprintf( stderr, "Error message: %s\n", Pa_GetErrorText( err ) );
    return err;

}