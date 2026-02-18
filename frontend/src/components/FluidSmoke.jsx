import React, { useRef, useEffect } from 'react';

const FluidSmoke = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let particles = [];
    
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    class SmokeParticle {
      constructor(x, y) {
        this.x = x;
        this.y = y;
        this.age = 0;
        this.maxAge = 100;
        this.size = Math.random() * 10 + 5;
        // Turbulence logic for the "swirly" movement
        this.vx = (Math.random() - 0.5) * 2;
        this.vy = (Math.random() - 0.5) * 2;
        this.rotation = Math.random() * Math.PI * 2;
        this.spin = (Math.random() - 0.5) * 0.1;
      }

      update() {
        this.x += this.vx;
        this.y += this.vy;
        this.rotation += this.spin;
        this.age++;
        this.size += 0.8; // Expands as it dissipates
        
        // Random "curvy" force
        this.vx += (Math.random() - 0.5) * 0.3;
        this.vy += (Math.random() - 0.5) * 0.3;
      }

      // --- Inside FluidSmoke.jsx -> draw() method ---

draw() {
  const opacity = 1 - (this.age / this.maxAge);
  ctx.save();
  ctx.translate(this.x, this.y);
  ctx.rotate(this.rotation);
  
  // Create a Radial Gradient for the Violet smoke
  const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, this.size);
  
  // 1. Core: Bright Violet (Matches your border: rgb(219, 17, 250))
  grad.addColorStop(0, `rgba(219, 17, 250, ${opacity * 0.4})`); 
  
  // 2. Mid: Deep Purple/Magenta for depth
  grad.addColorStop(0.5, `rgba(138, 43, 226, ${opacity * 0.15})`); 
  
  // 3. Edge: Fully transparent
  grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(0, 0, this.size, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}
    }

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach((p, i) => {
        p.update();
        p.draw();
        if (p.age > p.maxAge) particles.splice(i, 1);
      });
      requestAnimationFrame(animate);
    };

    const handleMouseMove = (e) => {
      // Add multiple particles for a thick trail
      for (let i = 0; i < 3; i++) {
        particles.push(new SmokeParticle(e.clientX, e.clientY));
      }
    };

    window.addEventListener('resize', resize);
    window.addEventListener('mousemove', handleMouseMove);
    resize();
    animate();

    return () => {
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        pointerEvents: 'none',
        zIndex: 1, // Sits behind the dashboard
      }}
    />
  );
};

export default FluidSmoke;