#ifndef INTERACTIONS_H
#define INTERACTIONS_H
#define W 2048
#define H 2048
#define DELTA 5 // pixel increment for arrow keys
#define TITLE_STRING "Torrential Rainbows Prototyping Sandbox"
int2 loc = {W/2, H/2};
int2 locplus = {0, 0};
bool dragMode = false; // mouse tracking mode

void keyboard(unsigned char key, int x, int y) {
  if (key == 'a') dragMode = !dragMode; //toggle tracking mode
  if (key == 27) exit(0);
  glutPostRedisplay();
}

void mouseMove(int x, int y) {
  if (dragMode) return;
  loc.x = x;
  loc.y = y;
  glutPostRedisplay();
}

void mouseDrag(int x, int y) {
  if (!dragMode) return;
  loc.x = x;
  loc.y = y;
  glutPostRedisplay();
}


void mouseUpDown(int button, int state, int x, int y){
  int diffx, diffy;
  if(state == GLUT_DOWN){
    locplus.x = x;
    locplus.y = y;
  }else{
    if(!dragMode) return;
    diffx = x - locplus.x;
    diffy = y - locplus.y;
    loc.x -= diffx;
    loc.y -= diffy;
    glutPostRedisplay();
  }
}



void handleSpecialKeypress(int key, int x, int y) {
  if (key == GLUT_KEY_LEFT)  loc.x -= DELTA;
  if (key == GLUT_KEY_RIGHT) loc.x += DELTA;
  if (key == GLUT_KEY_UP)    loc.y -= DELTA;
  if (key == GLUT_KEY_DOWN)  loc.y += DELTA;
  glutPostRedisplay();
}

void printInstructions() {
  printf("flashlight interactions\n");
  printf("a: toggle mouse tracking mode\n");
  printf("arrow keys: move ref location\n");
  printf("esc: close graphics window\n");
}

#endif
