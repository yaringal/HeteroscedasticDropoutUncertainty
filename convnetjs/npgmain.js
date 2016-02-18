//Simple game engine
//Author: Andrej Karpathy
//License: BSD
//This function does all the boring canvas_reg stuff. To use it, just create functions:
//update()          gets called every frame
//draw()            gets called every frame
//myinit()          gets called once in beginning
//mouseClick(x, y)  gets called on mouse click
//keyUp(keycode)    gets called when key is released
//keyDown(keycode)  gets called when key is pushed

var canvas_reg;
var ctx_reg;
var WIDTH;
var HEIGHT; 
var FPS;

function drawBubble(x, y, w, h, radius)
{
  var r = x + w;
  var b = y + h;
  ctx_reg.beginPath();
  ctx_reg.strokeStyle="black";
  ctx_reg.lineWidth="2";
  ctx_reg.moveTo(x+radius, y);
  ctx_reg.lineTo(x+radius/2, y-10);
  ctx_reg.lineTo(x+radius * 2, y);
  ctx_reg.lineTo(r-radius, y);
  ctx_reg.quadraticCurveTo(r, y, r, y+radius);
  ctx_reg.lineTo(r, y+h-radius);
  ctx_reg.quadraticCurveTo(r, b, r-radius, b);
  ctx_reg.lineTo(x+radius, b);
  ctx_reg.quadraticCurveTo(x, b, x, b-radius);
  ctx_reg.lineTo(x, y+radius);
  ctx_reg.quadraticCurveTo(x, y, x+radius, y);
  ctx_reg.stroke();
}

function drawRect(x, y, w, h){
  ctx_reg.beginPath();
  ctx_reg.rect(x,y,w,h);
  ctx_reg.closePath();
  ctx_reg.fill();
  ctx_reg.stroke();
}

function drawCircle(x, y, r){
  ctx_reg.beginPath();
  ctx_reg.arc(x, y, r, 0, Math.PI*2, true); 
  ctx_reg.closePath();
  ctx_reg.stroke();
  ctx_reg.fill();
}

//uniform distribution integer
function randi(s, e) {
  return Math.floor(Math.random()*(e-s) + s);
}

//uniform distribution
function randf(s, e) {
  return Math.random()*(e-s) + s;
}

//normal distribution random number
function randn(mean, variance) {
  var V1, V2, S;
  do {
    var U1 = Math.random();
    var U2 = Math.random();
    V1 = 2 * U1 - 1;
    V2 = 2 * U2 - 1;
    S = V1 * V1 + V2 * V2;
  } while (S > 1);
  X = Math.sqrt(-2 * Math.log(S) / S) * V1;
  X = mean + Math.sqrt(variance) * X;
  return X;
}

function eventClick(e) {
    
  //get position of cursor relative to top left of canvas_reg
  var x;
  var y;
  if (e.pageX || e.pageY) { 
    x = e.pageX;
    y = e.pageY;
  } else { 
    x = e.clientX + document.body.scrollLeft + document.documentElement.scrollLeft; 
    y = e.clientY + document.body.scrollTop + document.documentElement.scrollTop; 
  } 
  x -= canvas_reg.offsetLeft;
  y -= canvas_reg.offsetTop;

  //call user-defined callback
  mouseClick(x, y, e.shiftKey, e.ctrlKey);
}

var NPG_interval;

function NPGinit(FPS){
  //takes frames per secont to run at
  
  canvas_reg = document.getElementById('NPGcanvas');
  ctx_reg = canvas_reg.getContext('2d');
  WIDTH = canvas_reg.width;
  HEIGHT = canvas_reg.height;
  canvas_reg.addEventListener('click', eventClick, false);
  $('#NPGcanvas').on("tap", eventClick)
    
  NPG_interval = setInterval(NPGtick, 1000/FPS);
  
  myinit();
}

function NPGtick() {
    update_reg();
    draw_reg();
}
