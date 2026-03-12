const translations = {

en:{

nav_intro:"Introduction",
nav_team:"Team",
nav_project:"Project",
nav_results:"Results",

hero_title:"Unified Deep Learning Benchmark",
hero_subtitle:"Satellite Image Restoration & Generation",
hero_button:"Learn More",

intro_title:"Introduction",
intro_text1:"Satellite imagery plays an important role in environmental monitoring, agriculture, and disaster response.",
intro_text2:"However satellite images often suffer from cloud coverage, low resolution and noise.",
intro_text3:"Our project builds a unified benchmark for evaluating satellite image restoration and generation models.",

team_title:"Our Team",

royana_role:"Team Lead",
royana_text:"Developed the benchmarking pipeline and system architecture.",

nijat_role:"Dataset Integration",
nijat_text:"Implemented dataset preprocessing and normalization.",

pasha_role:"Dataset Analysis & Website Development",
pasha_text:"Analyzed datasets and worked on building the project website.",

huseyn_role:"Visualization & Website Development",
huseyn_text:"Created visualization tools and worked on website design.",

project_title:"Project Overview",
project_text:"The project develops a unified pipeline for benchmarking satellite image restoration models.",

results_title:"Results",
results_text:"Models are evaluated using PSNR, SSIM, SAM, ERGAS, QNR and FID metrics."

},

ru:{

nav_intro:"Введение",
nav_team:"Команда",
nav_project:"Проект",
nav_results:"Результаты",

hero_title:"Единый Бенчмарк Глубокого Обучения",
hero_subtitle:"Восстановление и Генерация Спутниковых Изображений",
hero_button:"Подробнее",

intro_title:"Введение",
intro_text1:"Спутниковые изображения используются в экологии, сельском хозяйстве и мониторинге катастроф.",
intro_text2:"Однако изображения часто имеют облака, низкое разрешение и шум.",
intro_text3:"Наш проект создаёт единый бенчмарк для оценки моделей улучшения спутниковых изображений.",

team_title:"Наша команда",

royana_role:"Руководитель команды",
nijat_role:"Интеграция данных",
pasha_role:"Анализ данных и разработка сайта",
huseyn_role:"Визуализация и разработка сайта",

project_title:"Описание проекта",
project_text:"Проект создаёт единый пайплайн для оценки моделей обработки спутниковых изображений.",

results_title:"Результаты",
results_text:"Модели оцениваются с помощью метрик PSNR, SSIM, SAM, ERGAS, QNR и FID."

},

az:{

nav_intro:"Giriş",
nav_team:"Komanda",
nav_project:"Layihə",
nav_results:"Nəticələr",

hero_title:"Birləşdirilmiş Deep Learning Benchmark",
hero_subtitle:"Peyk Şəkillərinin Bərpası və Generasiyası",
hero_button:"Ətraflı",

intro_title:"Giriş",
intro_text1:"Peyk şəkilləri kənd təsərrüfatı, ekologiya və fəlakət monitorinqində istifadə olunur.",
intro_text2:"Lakin peyk şəkillərində tez-tez buludlar, aşağı keyfiyyət və səs-küy olur.",
intro_text3:"Layihəmiz peyk şəkillərinin yaxşılaşdırılması modellərini qiymətləndirmək üçün vahid benchmark yaradır.",

team_title:"Komandamız",

royana_role:"Komanda rəhbəri",
nijat_role:"Dataset inteqrasiyası",
pasha_role:"Dataset analizi və veb sayt hazırlanması",
huseyn_role:"Vizualizasiya və veb sayt hazırlanması",

project_title:"Layihə haqqında",
project_text:"Layihə peyk şəkli bərpa modellərinin qiymətləndirilməsi üçün vahid pipeline yaradır.",

results_title:"Nəticələr",
results_text:"Modellər PSNR, SSIM, SAM, ERGAS, QNR və FID metrikaları ilə qiymətləndirilir."

}

};


const switcher = document.getElementById("languageSwitcher")

switcher.addEventListener("change",()=>{

const lang = switcher.value

document.querySelectorAll("[data-key]").forEach(element=>{

const key = element.getAttribute("data-key")

element.textContent = translations[lang][key]

})

})