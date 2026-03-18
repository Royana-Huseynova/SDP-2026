     1	/* ===== SMOOTH SCROLL ===== */
     2	document.querySelectorAll('.nav-links a').forEach(link => {
     3	    link.addEventListener('click', function (e) {
     4	        e.preventDefault();
     5	        const targetId = this.getAttribute('href');
     6	        const targetSection = document.querySelector(targetId);
     7	        if (targetSection) {
     8	            targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
     9	        }
    10	    });
    11	});
    12	
    13	/* ===== DYNAMIC SATELLITE BACKGROUND IMAGES ===== */
    14	(function () {
    15	    const layers = document.querySelectorAll('#dynamic-bg .bg-layer');
    16	    if (layers.length < 2) return;
    17	
    18	    const backgroundImages = [
    19	        'https://images-assets.nasa.gov/image/PIA12235/PIA12235~orig.jpg',
    20	        'https://images-assets.nasa.gov/image/GSFC_20171208_Archive_e001861/GSFC_20171208_Archive_e001861~orig.jpg',
    21	        'https://images-assets.nasa.gov/image/iss071e414090/iss071e414090~orig.jpg'
    22	    ];
    23	
    24	    let activeLayer = 0;
    25	    let imageIndex = 0;
    26	
    27	    function setImage(layerIndex, imageUrl) {
    28	        layers[layerIndex].style.backgroundImage = `url("${imageUrl}")`;
    29	    }
    30	
    31	    setImage(activeLayer, backgroundImages[imageIndex]);
    32	
    33	    setInterval(() => {
    34	        imageIndex = (imageIndex + 1) % backgroundImages.length;
    35	        const nextLayer = (activeLayer + 1) % layers.length;
    36	
    37	        setImage(nextLayer, backgroundImages[imageIndex]);
    38	        layers[nextLayer].classList.add('active');
    39	        layers[activeLayer].classList.remove('active');
    40	
    41	        activeLayer = nextLayer;
    42	    }, 7000);
    43	})();
    44	
    45	/* ===== CANVAS SPACE BACKGROUND ===== */
    46	(function () {
    47	    const canvas = document.getElementById('bg-canvas');
    48	    const ctx = canvas.getContext('2d');
    49	
    50	    let W, H;
    51	
    52	    function resize() {
    53	        W = canvas.width  = window.innerWidth;
    54	        H = canvas.height = window.innerHeight;
    55	    }
    56	    window.addEventListener('resize', resize);
    57	    resize();
    58	
    59	    /* --- Stars --- */
    60	    const STAR_COUNT = 280;
    61	    const stars = Array.from({ length: STAR_COUNT }, () => ({
    62	        x: Math.random() * 1,
    63	        y: Math.random() * 1,
    64	        r: Math.random() * 1.4 + 0.3,
    65	        alpha: Math.random() * 0.6 + 0.3,
    66	        twinkleSpeed: Math.random() * 0.008 + 0.002,
    67	        twinkleOffset: Math.random() * Math.PI * 2
    68	    }));
    69	
    70	    /* --- Shooting stars --- */
    71	    const shooters = [];
    72	    function spawnShooter() {
    73	        shooters.push({
    74	            x: Math.random() * W,
    75	            y: Math.random() * H * 0.5,
    76	            len: Math.random() * 120 + 60,
    77	            speed: Math.random() * 6 + 4,
    78	            angle: Math.PI / 5,
    79	            alpha: 1,
    80	            life: 0
    81	        });
    82	    }
    83	    setInterval(spawnShooter, 3200);
    84	
    85	    /* --- Satellites --- */
    86	    function makeSatellite(cx, cy, orbitA, orbitB, speed, angleDeg, tiltDeg) {
    87	        return {
    88	            cx, cy,
    89	            orbitA, orbitB,
    90	            speed,
    91	            angle: (angleDeg * Math.PI) / 180,
    92	            tilt:  (tiltDeg  * Math.PI) / 180
    93	        };
    94	    }
    95	
    96	    const satellites = [
    97	        makeSatellite(0.38, 0.42, 0.28, 0.14, 0.0004, 0,    15),
    98	        makeSatellite(0.68, 0.60, 0.18, 0.10, 0.0007, 90,  -10),
    99	        makeSatellite(0.20, 0.72, 0.14, 0.07, 0.0010, 200,  25),
   100	        makeSatellite(0.80, 0.25, 0.12, 0.06, 0.0013, 310, -20),
   101	    ];
   102	
   103	    /* Draw a satellite at canvas coords (sx, sy), rotated by heading angle */
   104	    function drawSatellite(sx, sy, heading, scale) {
   105	        const s = scale || 1;
   106	        ctx.save();
   107	        ctx.translate(sx, sy);
   108	        ctx.rotate(heading);
   109	        ctx.scale(s, s);
   110	
   111	        /* Body */
   112	        ctx.fillStyle = '#c8d6f0';
   113	        ctx.fillRect(-10, -6, 20, 12);
   114	
   115	        /* Centre stripe */
   116	        ctx.fillStyle = '#1e3a8a';
   117	        ctx.fillRect(-3, -6, 6, 12);
   118	
   119	        /* Solar panels left */
   120	        ctx.fillStyle = '#1d4ed8';
   121	        ctx.fillRect(-36, -4, 22, 8);
   122	        /* Panel frame left */
   123	        ctx.strokeStyle = '#60a5fa';
   124	        ctx.lineWidth = 0.8;
   125	        ctx.strokeRect(-36, -4, 22, 8);
   126	        /* Panel grid left */
   127	        for (let gi = 1; gi < 4; gi++) {
   128	            ctx.beginPath();
   129	            ctx.moveTo(-36 + gi * (22 / 4), -4);
   130	            ctx.lineTo(-36 + gi * (22 / 4),  4);
   131	            ctx.stroke();
   132	        }
   133	        ctx.beginPath(); ctx.moveTo(-36, 0); ctx.lineTo(-14, 0); ctx.stroke();
   134	
   135	        /* Solar panels right */
   136	        ctx.fillStyle = '#1d4ed8';
   137	        ctx.fillRect(14, -4, 22, 8);
   138	        ctx.strokeStyle = '#60a5fa';
   139	        ctx.lineWidth = 0.8;
   140	        ctx.strokeRect(14, -4, 22, 8);
   141	        for (let gi = 1; gi < 4; gi++) {
   142	            ctx.beginPath();
   143	            ctx.moveTo(14 + gi * (22 / 4), -4);
   144	            ctx.lineTo(14 + gi * (22 / 4),  4);
   145	            ctx.stroke();
   146	        }
   147	        ctx.beginPath(); ctx.moveTo(14, 0); ctx.lineTo(36, 0); ctx.stroke();
   148	
   149	        /* Antenna dish */
   150	        ctx.strokeStyle = '#93c5fd';
   151	        ctx.lineWidth = 1;
   152	        ctx.beginPath();
   153	        ctx.arc(0, -11, 5, Math.PI, 0);
   154	        ctx.stroke();
   155	        ctx.beginPath();
   156	        ctx.moveTo(0, -6);
   157	        ctx.lineTo(0, -11);
   158	        ctx.stroke();
   159	
   160	        /* Signal beam */
   161	        ctx.strokeStyle = 'rgba(96,165,250,0.25)';
   162	        ctx.lineWidth = 1;
   163	        ctx.beginPath();
   164	        ctx.moveTo(0, -16);
   165	        ctx.lineTo(-10, -30);
   166	        ctx.moveTo(0, -16);
   167	        ctx.lineTo(0,  -32);
   168	        ctx.moveTo(0, -16);
   169	        ctx.lineTo(10, -30);
   170	        ctx.stroke();
   171	
   172	        ctx.restore();
   173	    }
   174	
   175	    /* Draw an elliptical orbit path */
   176	    function drawOrbit(cx, cy, a, b, tilt, alpha) {
   177	        ctx.save();
   178	        ctx.translate(cx, cy);
   179	        ctx.rotate(tilt);
   180	        ctx.scale(1, b / a);
   181	        ctx.beginPath();
   182	        ctx.arc(0, 0, a, 0, Math.PI * 2);
   183	        ctx.restore();
   184	        ctx.strokeStyle = `rgba(96,165,250,${alpha})`;
   185	        ctx.lineWidth = 0.6;
   186	        ctx.setLineDash([6, 10]);
   187	        ctx.stroke();
   188	        ctx.setLineDash([]);
   189	    }
   190	
   191	    let t = 0;
   192	
   193	    function draw() {
   194	        t++;
   195	
   196	        /* Deep space gradient background */
   197	        const grad = ctx.createLinearGradient(0, 0, W, H);
   198	        grad.addColorStop(0,   'rgba(6, 13, 31, 0.54)');
   199	        grad.addColorStop(0.5, 'rgba(10, 22, 40, 0.48)');
   200	        grad.addColorStop(1,   'rgba(6, 13, 31, 0.54)');
   201	        ctx.fillStyle = grad;
   202	        ctx.fillRect(0, 0, W, H);
   203	
   204	        /* Nebula glow blobs */
   205	        const nebulaData = [
   206	            { x: 0.15, y: 0.2,  r: 0.22, c1: 'rgba(37,99,235,0.06)',  c2: 'rgba(0,0,0,0)' },
   207	            { x: 0.75, y: 0.65, r: 0.20, c1: 'rgba(79,70,229,0.05)',  c2: 'rgba(0,0,0,0)' },
   208	            { x: 0.50, y: 0.85, r: 0.18, c1: 'rgba(14,165,233,0.05)', c2: 'rgba(0,0,0,0)' },
   209	        ];
   210	        nebulaData.forEach(n => {
   211	            const rg = ctx.createRadialGradient(n.x * W, n.y * H, 0, n.x * W, n.y * H, n.r * W);
   212	            rg.addColorStop(0, n.c1);
   213	            rg.addColorStop(1, n.c2);
   214	            ctx.fillStyle = rg;
   215	            ctx.fillRect(0, 0, W, H);
   216	        });
   217	
   218	        /* Stars */
   219	        stars.forEach(s => {
   220	            const twinkle = 0.5 + 0.5 * Math.sin(t * s.twinkleSpeed + s.twinkleOffset);
   221	            ctx.beginPath();
   222	            ctx.arc(s.x * W, s.y * H, s.r, 0, Math.PI * 2);
   223	            ctx.fillStyle = `rgba(255,255,255,${s.alpha * twinkle})`;
   224	            ctx.fill();
   225	        });
   226	
   227	        /* Shooting stars */
   228	        for (let i = shooters.length - 1; i >= 0; i--) {
   229	            const sh = shooters[i];
   230	            sh.x += Math.cos(sh.angle) * sh.speed;
   231	            sh.y += Math.sin(sh.angle) * sh.speed;
   232	            sh.life++;
   233	            sh.alpha = Math.max(0, 1 - sh.life / 40);
   234	
   235	            const tailX = sh.x - Math.cos(sh.angle) * sh.len;
   236	            const tailY = sh.y - Math.sin(sh.angle) * sh.len;
   237	            const sg = ctx.createLinearGradient(tailX, tailY, sh.x, sh.y);
   238	            sg.addColorStop(0, 'rgba(255,255,255,0)');
   239	            sg.addColorStop(1, `rgba(255,255,255,${sh.alpha})`);
   240	            ctx.beginPath();
   241	            ctx.moveTo(tailX, tailY);
   242	            ctx.lineTo(sh.x, sh.y);
   243	            ctx.strokeStyle = sg;
   244	            ctx.lineWidth = 1.5;
   245	            ctx.stroke();
   246	
   247	            if (sh.alpha <= 0) shooters.splice(i, 1);
   248	        }
   249	
   250	        /* Orbit paths and satellites */
   251	        satellites.forEach(sat => {
   252	            sat.angle += sat.speed;
   253	
   254	            const cx = sat.cx * W;
   255	            const cy = sat.cy * H;
   256	            const a  = sat.orbitA * Math.min(W, H);
   257	            const b  = sat.orbitB * Math.min(W, H);
   258	
   259	            drawOrbit(cx, cy, a, b, sat.tilt, 0.12);
   260	
   261	            /* Position on tilted ellipse */
   262	            const localX = a * Math.cos(sat.angle);
   263	            const localY = b * Math.sin(sat.angle);
   264	            const sx = cx + localX * Math.cos(sat.tilt) - localY * Math.sin(sat.tilt);
   265	            const sy = cy + localX * Math.sin(sat.tilt) + localY * Math.cos(sat.tilt);
   266	
   267	            /* Heading tangent */
   268	            const nextAngle = sat.angle + 0.01;
   269	            const nx2 = cx + a * Math.cos(nextAngle) * Math.cos(sat.tilt) - b * Math.sin(nextAngle) * Math.sin(sat.tilt);
   270	            const ny2 = cy + a * Math.cos(nextAngle) * Math.sin(sat.tilt) + b * Math.sin(nextAngle) * Math.cos(sat.tilt);
   271	            const heading = Math.atan2(ny2 - sy, nx2 - sx);
   272	
   273	            /* Glow under satellite */
   274	            const glowR = ctx.createRadialGradient(sx, sy, 0, sx, sy, 24);
   275	            glowR.addColorStop(0, 'rgba(96,165,250,0.18)');
   276	            glowR.addColorStop(1, 'rgba(96,165,250,0)');
   277	            ctx.fillStyle = glowR;
   278	            ctx.beginPath();
   279	            ctx.arc(sx, sy, 24, 0, Math.PI * 2);
   280	            ctx.fill();
   281	
   282	            drawSatellite(sx, sy, heading, 0.85);
   283	        });
   284	
   285	        requestAnimationFrame(draw);
   286	    }
   287	
   288	    draw();
   289	})();

