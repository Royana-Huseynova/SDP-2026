// ===============================
// Satellite Benchmark Website JS
// Base script file (no changes to layout)
// ===============================

document.addEventListener("DOMContentLoaded", function () {

    // Smooth scroll for navbar links

    const links = document.querySelectorAll('nav a');

    links.forEach(function(link) {

        link.addEventListener("click", function(e) {

            const targetId = this.getAttribute("href");

            if (targetId.startsWith("#")) {

                e.preventDefault();

                const target = document.querySelector(targetId);

                if (target) {

                    target.scrollIntoView({
                        behavior: "smooth"
                    });

                }

            }

        });

    });


    // Future scripts can be added here safely

});
