const canvas = document.getElementById("bg-canvas");
const ctx = canvas ? canvas.getContext("2d") : null;

let particles = [];
let mouse = { x: null, y: null, radius: 120 };

function resizeCanvas() {
    if (!canvas || !ctx) return;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    createParticles();
}

function createParticles() {
    if (!canvas) return;

    particles = [];
    const count = Math.max(60, Math.floor((canvas.width * canvas.height) / 18000));

    for (let i = 0; i < count; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            size: Math.random() * 1.8 + 0.6,
            speedX: (Math.random() - 0.5) * 0.35,
            speedY: (Math.random() - 0.5) * 0.35,
            opacity: Math.random() * 0.45 + 0.2
        });
    }
}

function drawParticles() {
    if (!canvas || !ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < particles.length; i++) {
        const p = particles[i];

        p.x += p.speedX;
        p.y += p.speedY;

        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        let sizeBoost = 0;
        if (mouse.x !== null && mouse.y !== null) {
            const dx = p.x - mouse.x;
            const dy = p.y - mouse.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < mouse.radius) {
                sizeBoost = (mouse.radius - dist) / mouse.radius;
            }
        }

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size + sizeBoost * 1.2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(147, 197, 253, ${p.opacity})`;
        ctx.fill();
    }

    connectParticles();
    requestAnimationFrame(drawParticles);
}

function connectParticles() {
    if (!canvas || !ctx) return;

    for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < 110) {
                const opacity = (1 - dist / 110) * 0.12;
                ctx.beginPath();
                ctx.moveTo(particles[i].x, particles[i].y);
                ctx.lineTo(particles[j].x, particles[j].y);
                ctx.strokeStyle = `rgba(147, 197, 253, ${opacity})`;
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }
    }
}

window.addEventListener("mousemove", (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
});

window.addEventListener("mouseleave", () => {
    mouse.x = null;
    mouse.y = null;
});

window.addEventListener("resize", resizeCanvas);

resizeCanvas();
drawParticles();

/* ================= BACKGROUND IMAGE ROTATION ================= */

const bgLayers = document.querySelectorAll("#dynamic-bg .bg-layer");

const bgImages = [
    "images/bg1.jpg",
    "images/bg2.jpg",
    "images/bg3.jpg"
];

let currentBg = 0;
let activeLayer = 0;

function preloadBackgrounds(images) {
    images.forEach((src) => {
        const img = new Image();
        img.src = src;
    });
}

function setBackgroundImage(layer, src) {
    layer.style.backgroundImage = `
        linear-gradient(rgba(2, 6, 23, 0.72), rgba(2, 6, 23, 0.82)),
        url('${src}')
    `;
    layer.style.backgroundSize = "cover";
    layer.style.backgroundPosition = "center";
    layer.style.backgroundRepeat = "no-repeat";
}

function rotateBackgrounds() {
    if (!bgLayers.length || bgImages.length === 0) return;

    const nextLayer = activeLayer === 0 ? 1 : 0;
    currentBg = (currentBg + 1) % bgImages.length;

    setBackgroundImage(bgLayers[nextLayer], bgImages[currentBg]);
    bgLayers[nextLayer].classList.add("active");
    bgLayers[activeLayer].classList.remove("active");

    activeLayer = nextLayer;
}

if (bgLayers.length > 0 && bgImages.length > 0) {
    preloadBackgrounds(bgImages);
    setBackgroundImage(bgLayers[0], bgImages[0]);

    if (bgLayers[1]) {
        setBackgroundImage(bgLayers[1], bgImages[1 % bgImages.length]);
        bgLayers[1].classList.remove("active");
    }

    setInterval(rotateBackgrounds, 9000);
}

/* ================= ACTIVE NAV LINK ================= */

const sections = document.querySelectorAll("section[id]");
const navLinks = document.querySelectorAll(".nav-links a");

function setActiveNav() {
    let currentSectionId = "";

    sections.forEach((section) => {
        const rect = section.getBoundingClientRect();
        if (rect.top <= 140 && rect.bottom >= 140) {
            currentSectionId = section.id;
        }
    });

    navLinks.forEach((link) => {
        link.classList.remove("active");
        const href = link.getAttribute("href");
        if (href === `#${currentSectionId}`) {
            link.classList.add("active");
        }
    });
}

window.addEventListener("scroll", setActiveNav);
window.addEventListener("load", setActiveNav);

/* ================= SMOOTH SCROLL OFFSET FOR FIXED NAV ================= */

navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
        const targetId = link.getAttribute("href");
        if (!targetId || !targetId.startsWith("#")) return;

        const target = document.querySelector(targetId);
        if (!target) return;

        e.preventDefault();

        const nav = document.querySelector(".navbar");
        const navHeight = nav ? nav.offsetHeight : 0;
        const targetTop = target.getBoundingClientRect().top + window.scrollY - navHeight - 20;

        window.scrollTo({
            top: targetTop,
            behavior: "smooth"
        });
    });
});

