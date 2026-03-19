// Dynamic subtle particle behavior for the background context
const background = document.querySelector('.satellite-grid');
const starCount = 40;

for (let i = 0; i < starCount; i++) {
  const star = document.createElement('span');
  const size = Math.random() * 3 + 0.8;
  star.style.position = 'absolute';
  star.style.width = `${size}px`;
  star.style.height = `${size}px`;
  star.style.borderRadius = '50%';
  star.style.background = 'rgba(255, 255, 255, 0.65)';
  star.style.left = `${Math.random() * 100}%`;
  star.style.top = `${Math.random() * 100}%`;
  star.style.filter = `blur(${Math.random() * 1.2}px)`;
  star.style.opacity = Math.random() * 0.85;
  star.style.animation = `twinkle ${Math.random() * 8 + 6}s infinite ease-in-out`;
  background.appendChild(star);
}

const style = document.createElement('style');
style.textContent = `@keyframes twinkle { 0%, 100% { opacity: 0.2; } 50% { opacity: 0.95; } }`;
document.head.appendChild(style);

// Image Lightbox Modal
const modal = document.getElementById('imageModal');
const closeBtn = document.querySelector('.close-btn');
const clickableImages = document.querySelectorAll('.clickable-image');
const modalImage = document.querySelector('.modal-image');

// Open modal when image is clicked
clickableImages.forEach(img => {
  img.addEventListener('click', function() {
    modal.classList.add('show');
    modalImage.src = this.src;
    modalImage.alt = this.alt;
    document.body.style.overflow = 'hidden';
  });
});

// Close modal
function closeModal() {
  modal.classList.remove('show');
  document.body.style.overflow = 'auto';
}

closeBtn.addEventListener('click', closeModal);

// Close modal when clicking outside the image
modal.addEventListener('click', function(event) {
  if (event.target === modal) {
    closeModal();
  }
});

// Close modal on Escape key
document.addEventListener('keydown', function(event) {
  if (event.key === 'Escape') {
    closeModal();
  }
});

