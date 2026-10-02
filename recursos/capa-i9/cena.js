/* Maquete original i9. Three.js fica incorporado ao HTML; nenhum recurso remoto. */
(()=>{
 'use strict';
 const canvas=document.getElementById('diorama-canvas'),panel=document.getElementById('diorama-panel'),cover=panel.closest('.cover');
 const loading=document.getElementById('diorama-loading'),pauseButton=document.getElementById('scene-pause'),resetButton=document.getElementById('scene-reset');
 let renderer;
 try{renderer=new THREE.WebGLRenderer({canvas,alpha:true,antialias:true,powerPreference:'low-power',preserveDrawingBuffer:true});}
 catch(error){panel.classList.add('no-webgl');loading.hidden=true;pauseButton.hidden=true;resetButton.hidden=true;return;}
 renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.7));
 renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.12;
 renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.VSMShadowMap;renderer.setClearColor(0x141415,0);
 const scene=new THREE.Scene(),world=new THREE.Group();scene.add(world);
 const camera=new THREE.OrthographicCamera(-8,8,7,-7,.1,80);
 const target=new THREE.Vector3(0,1,0);let yaw=.73,pitch=.60,desiredYaw=yaw,desiredPitch=pitch;
 const materials={};
 const palette={red:0xbf0001,redLight:0xe22429,white:0xf1f0eb,wall:0xbcbdbc,concrete:0x919292,dark:0x242a30,steel:0x73808c,metal:0xb7bfc3,black:0x151a20,orange:0xf39a36,reflective:0xf6f6d4,navy:0x304253,skin:0xcb9678,yellow:0xe6b34c,wood:0xba8f60,plant:0x576452};
 function mat(key){if(materials[key])return materials[key];return materials[key]=new THREE.MeshStandardMaterial({color:palette[key]??key,roughness:key==='metal'?.38:.8,metalness:key==='metal'?.45:key==='steel'?.25:0});}
 const boxGeometry=new THREE.BoxGeometry(1,1,1),sphereGeometry=new THREE.SphereGeometry(1,20,14),cylinderGeometry=new THREE.CylinderGeometry(1,1,1,24);
 function mesh(geometry,material,parent=world){const m=new THREE.Mesh(geometry,material?.isMaterial?material:mat(material));m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;}
 function box(x,y,z,w,h,d,key,parent=world){const m=mesh(boxGeometry,key,parent);m.position.set(x,y,z);m.scale.set(w,h,d);return m;}
 function sphere(x,y,z,rx,ry,rz,key,parent=world){const m=mesh(sphereGeometry,key,parent);m.position.set(x,y,z);m.scale.set(rx,ry,rz);return m;}
 function cylinder(x,y,z,r,h,key,parent=world){const m=mesh(cylinderGeometry,key,parent);m.position.set(x,y,z);m.scale.set(r,h,r);return m;}
 function group(x,y,z,parent=world){const g=new THREE.Group();g.position.set(x,y,z);parent.add(g);return g;}
 function bar(a,b,r,key,parent=world){const av=new THREE.Vector3(...a),bv=new THREE.Vector3(...b),v=bv.clone().sub(av);const m=cylinder(0,0,0,r,v.length(),key,parent);m.position.copy(av.add(bv).multiplyScalar(.5));m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),v.normalize());return m;}
 function roundedBox(w,h,d,r,key,parent=world){const shape=new THREE.Shape(),x=-w/2,y=-d/2,bevel=Math.min(.04,h/4,r/2);shape.moveTo(x+r,y);shape.lineTo(x+w-r,y);shape.quadraticCurveTo(x+w,y,x+w,y+r);shape.lineTo(x+w,y+d-r);shape.quadraticCurveTo(x+w,y+d,x+w-r,y+d);shape.lineTo(x+r,y+d);shape.quadraticCurveTo(x,y+d,x,y+d-r);shape.lineTo(x,y+r);shape.quadraticCurveTo(x,y,x+r,y);const geom=new THREE.ExtrudeGeometry(shape,{depth:h-2*bevel,bevelEnabled:true,bevelSegments:2,steps:1,bevelSize:bevel,bevelThickness:bevel,curveSegments:8});geom.rotateX(-Math.PI/2);geom.translate(0,-h/2+bevel,0);return mesh(geom,key,parent);}
 function label(text,w,h,{bg='#202429',color='#ffffff',size=64,bold=true}={},parent=world){const c=document.createElement('canvas');c.width=512;c.height=Math.round(512*h/w);const ctx=c.getContext('2d');ctx.fillStyle=bg;ctx.fillRect(0,0,c.width,c.height);ctx.fillStyle=color;ctx.textAlign='center';ctx.textBaseline='middle';ctx.font=`${bold?'700':'500'} ${size}px Arial`;ctx.fillText(text,256,c.height/2);const texture=new THREE.CanvasTexture(c);texture.colorSpace=THREE.SRGBColorSpace;texture.anisotropy=4;const m=mesh(new THREE.PlaneGeometry(w,h),new THREE.MeshBasicMaterial({map:texture}),parent);m.castShadow=false;return m;}
 // Studio lighting, soft directional shadows, and a physical model plinth.
 scene.add(new THREE.HemisphereLight(0xe4edf6,0x473833,2.5));
 const key=new THREE.DirectionalLight(0xffefde,3.2);key.position.set(-4,14,6);key.castShadow=true;key.shadow.mapSize.set(2048,2048);Object.assign(key.shadow.camera,{left:-9,right:9,top:9,bottom:-9,near:.5,far:35});key.shadow.normalBias=.035;key.shadow.bias=-.0002;key.shadow.radius=4;key.shadow.blurSamples=8;scene.add(key);
 const fill=new THREE.DirectionalLight(0xd9e5ff,1.4);fill.position.set(8,6,-4);scene.add(fill);
 const ground=new THREE.Mesh(new THREE.PlaneGeometry(200,200),new THREE.ShadowMaterial({opacity:.22}));ground.rotation.x=-Math.PI/2;ground.position.y=-.65;ground.receiveShadow=true;scene.add(ground);
 const plinth=roundedBox(10.5,.44,8,.25,'dark');plinth.position.y=-.25;
 const edge=roundedBox(10.52,.045,8.02,.24,'red');edge.position.y=-.07;
 const floor=roundedBox(10.35,.16,7.85,.21,'concrete');floor.position.y=.025;
 const modelTitle=label('i9   /   ENGENHARIA E TECNOLOGIA',4,.22,{bg:'#242a30',color:'#e6e5e6',size:26});modelTitle.position.set(-.5,-.25,4.025);
 // Concrete expansion joints and protected pedestrian route.
 for(let x=-4;x<=4;x+=2)box(x,.112,0,.014,.008,7.65,0x7b7e80);
 for(let z=-3;z<=3;z+=2)box(0,.112,z,10.1,.008,.014,0x7b7e80);
 box(0,.122,2.55,9.5,.018,1.05,0x646c70);
 box(0,.136,2,9.5,.015,.035,'yellow');box(0,.136,3.1,9.5,.015,.035,'yellow');
 for(let x=-4.3;x<4.4;x+=.8)box(x,.139,2.56,.36,.013,.045,'white');
 for(let z=.72;z<2;z+=.23)box(1.3,.14,z,.83,.012,.12,'white');
 const walkSign=label('PEDESTRES',1.6,.32,{bg:'#646c70',color:'#f4f4ed',size:62});walkSign.rotation.x=-Math.PI/2;walkSign.position.set(2.7,.15,2.57);
 // Cutaway factory: open front, corrugated walls, red structural frame.
 const factory=group(-.5,.13,-1.9);
 box(0,.055,0,6.6,.12,3.2,0xd0d0c9,factory);
 box(0,1.65,-1.57,6.6,3.3,.16,'wall',factory);box(-3.25,1.05,0,.16,2.1,3.15,'wall',factory);
 for(let x=-3.15;x<3.2;x+=.2)box(x,1.47,-1.46,.025,2.9,.025,0xa5a9aa,factory);
 for(let z=-1.4;z<1.5;z+=.2)box(-3.15,1.04,z,.025,2,.025,0xcdd0cd,factory);
 // Four steel columns with base plates and bolted joints.
 for(const x of [-3.25,3.25])for(const z of [-1.5,1.5]){box(x,.06,z,.32,.12,.32,'steel',factory);box(x,1.69,z,.15,3.35,.15,'red',factory);for(const dx of [-.12,.12])for(const dz of [-.12,.12])cylinder(x+dx,.14,z+dz,.025,.035,'metal',factory);}
 box(0,3.31,1.5,6.7,.23,.18,'red',factory);box(0,3.31,-1.5,6.7,.23,.18,'red',factory);
 box(-3.25,3.31,0,.18,.23,3.2,'red',factory);box(3.25,3.31,0,.18,.23,3.2,'red',factory);
 // Rear roof strip leaves the machinery visible, like an architectural cutaway.
 box(0,3.47,-1.1,6.85,.13,1.2,'white',factory);for(let x=-3.35;x<=3.4;x+=.22)box(x,3.55,-1.1,.033,.06,1.2,0xcbd0d1,factory);
 for(const x of [-2.8,2.8])bar([x,3.2,-1.4],[x,2.6,-.7],.035,'metal',factory);
 const brandPanel=box(.35,3.28,1.615,2.55,.57,.055,0x16191e,factory);
 const brand=label('i9',.86,.34,{bg:'#16191e',color:'#ffffff',size:190},factory);brand.position.set(-.39,3.29,1.652);
 const officialLogo=document.querySelector('.brand-logo');
 function paintLogo(){if(!officialLogo.complete||!officialLogo.naturalWidth)return;const c=document.createElement('canvas');c.width=512;c.height=240;const ct=c.getContext('2d');ct.fillStyle='#16191e';ct.fillRect(0,0,512,240);const ratio=officialLogo.naturalWidth/officialLogo.naturalHeight;const iw=380,ih=iw/ratio;ct.drawImage(officialLogo,(512-iw)/2,(240-ih)/2,iw,ih);const tex=new THREE.CanvasTexture(c);tex.colorSpace=THREE.SRGBColorSpace;brand.material.map.dispose();brand.material.map=tex;brand.material.needsUpdate=true;renderStill();}
 const brandWords=label('ENGENHARIA',1.25,.19,{bg:'#16191e',color:'#f1eeee',size:58},factory);brandWords.position.set(.73,3.38,1.653);
 const brandWords2=label('E TECNOLOGIA',1.25,.19,{bg:'#16191e',color:'#bfb8bc',size:50},factory);brandWords2.position.set(.73,3.16,1.653);
 for(const x of [-1.75,.15,2.05]){box(x,2.6,-1.45,1.65,.66,.05,0x536c7a,factory);box(x,2.6,-1.405,.035,.67,.025,'metal',factory);box(x,2.6,-1.405,1.65,.035,.025,'metal',factory);}
 // Cable conduit and lamps along the rear wall.
 bar([-2.85,2,-1.36],[2.9,2,-1.36],.035,'steel',factory);bar([-2.85,2,-1.36],[-2.85,.5,-1.36],.035,'steel',factory);
 const lampMat=new THREE.MeshStandardMaterial({color:0xfff5d9,emissive:0xffead3,emissiveIntensity:.6});
 for(const x of [-1.9,1.7])box(x,3.15,-.5,1.05,.04,.13,lampMat,factory);
 // Electrical cabinet and lathe, with tangible knobs and an instrument panel.
 box(-2.5,.82,-.76,.65,1.52,.62,'white',factory);box(-2.5,.82,-.438,.54,1.3,.035,'metal',factory);box(-2.26,.88,-.402,.03,.19,.05,'dark',factory);
 const voltage=label('⚡',.22,.25,{bg:'#e6b34c',color:'#242a30',size:200},factory);voltage.position.set(-2.5,1.1,-.411);
 const machine=group(.2,.08,-.55,factory);box(0,.58,0,2.55,1.16,.9,0x667b85,machine);box(0,1.19,0,2.68,.12,1.02,'metal',machine);
 box(-.85,1.62,-.03,.5,.74,.78,'white',machine);box(.78,1.56,-.03,.49,.63,.72,'white',machine);
 const spindle=cylinder(-.18,1.58,-.03,.18,1.15,'metal',machine);spindle.rotation.z=Math.PI/2;
 box(.89,1.38,.42,.49,.31,.07,'dark',machine);box(.78,1.43,.462,.17,.12,.01,0x72c6c9,machine);sphere(1.02,1.4,.48,.045,.045,.025,'red',machine);
 // Conveyor in the front-left production area.
 const conveyor=group(-1.95,.15,.45);box(0,.72,0,3.25,.14,.86,'dark',conveyor);
 for(const x of [-1.35,1.35])for(const z of [-.3,.3])box(x,.37,z,.09,.72,.09,'steel',conveyor);
 box(0,.79,-.46,3.45,.12,.09,'metal',conveyor);box(0,.79,.46,3.45,.12,.09,'metal',conveyor);
 const rollers=[];for(let x=-1.5;x<=1.55;x+=.18){const roll=cylinder(x,.79,0,.07,.8,'steel',conveyor);roll.rotation.x=Math.PI/2;rollers.push(roll);}
 function parcel(parent,x=0,y=0,z=0,size=.5){const p=group(x,y,z,parent);box(0,size*.38,0,size,size*.76,size*.8,'wood',p);box(0,size*.764,0,size*.14,.012,size*.8,0xdfc896,p);box(0,size*.37,size*.407,size*.14,size*.75,.008,0xdcc69e,p);const tag=label('i9',size*.3,size*.18,{bg:'#e5ddc8',color:'#272a2c',size:170},p);tag.position.set(-size*.23,size*.4,size*.408);return p;}
 const parcels=[parcel(conveyor,-1,.87,0,.49),parcel(conveyor,.2,.87,0,.49)];
 // Receiving pallet, outside the pedestrian lane.
 for(const z of [-.3,0,.3])box(3.73,.18,z+.38,.95,.09,.18,'wood');for(const x of [3.36,4.1])box(x,.25,.38,.13,.07,.87,'wood');
 parcel(world,3.52,.29,.2,.46);parcel(world,3.96,.29,.2,.42);parcel(world,3.63,.29,.64,.46);parcel(world,3.72,.66,.4,.42);
 // Forklift in i9 red, with tyres, roll cage, mast, seat, and forks.
 const forklift=group(-3.73,.17,1.45);forklift.rotation.y=-Math.PI/2;
 const liftBody=roundedBox(.81,.5,1.3,.13,'red',forklift);liftBody.position.y=.43;box(0,.6,-.3,.79,.22,.55,'redLight',forklift);
 for(const x of [-.44,.44])for(const z of [-.45,.43]){const tyre=cylinder(x,.3,z,.24,.14,'black',forklift);tyre.rotation.z=Math.PI/2;const hub=cylinder(x*1.02,.3,z,.115,.15,'metal',forklift);hub.rotation.z=Math.PI/2;}
 for(const x of [-.34,.34])for(const z of [-.43,.3])box(x,1.14,z,.045,1.05,.045,'dark',forklift);
 box(0,1.69,-.07,.88,.075,.93,'dark',forklift);for(const z of [-.31,-.1,.11])box(0,1.74,z,.82,.04,.04,'metal',forklift);
 box(0,.88,-.12,.43,.18,.47,'black',forklift);box(0,1.09,-.31,.45,.4,.1,'black',forklift);
 for(const x of [-.28,.28]){box(x,1,.7,.075,1.64,.09,'dark',forklift);box(x,.22,1.02,.12,.06,.68,'steel',forklift);}
 box(0,.5,.75,.74,.12,.07,'steel',forklift);const liftLabel=label('i9',.4,.2,{bg:'#bf0001',color:'#fff',size:200},forklift);liftLabel.position.set(0,.59,-.657);liftLabel.rotation.y=Math.PI;
 // Human miniatures with hard-hats, ear protection, reflective vests and boots.
 function worker(x,z,angle=0,helmet='white'){
  const p=group(x,.15,z);p.rotation.y=angle;
  const hips=group(0,.58,0,p),legs=[];
  for(const side of [-1,1]){const l=group(side*.10,0,0,hips);box(0,-.21,0,.135,.42,.15,'navy',l);box(0,-.445,.04,.16,.11,.24,'black',l);legs.push(l);}
  const torso=mesh(new THREE.CylinderGeometry(.18,.15,.4,8),'orange',p);torso.position.y=.81;torso.scale.z=.75;
  // Reflective bands wrap all around the vest, not just a front decal.
  const band=mesh(new THREE.CylinderGeometry(.179,.174,.055,8),'reflective',p);band.position.y=.79;band.scale.z=.765;
  for(const side of [-1,1])for(const front of [-1,1])box(side*.09,.89,front*.123,.035,.23,.013,'reflective',p);
  const arms=[];for(const side of [-1,1]){const a=group(side*.21,.94,0,p);const sleeve=mesh(new THREE.CapsuleGeometry(.065,.23,4,8),'navy',a);sleeve.position.y=-.14;sphere(0,-.32,0,.055,.07,.055,'skin',a);arms.push(a);}
  cylinder(0,1.05,0,.06,.11,'skin',p);sphere(0,1.18,0,.118,.15,.108,'skin',p);
  const hat=group(0,1.25,0,p);
  const hardhat=mesh(new THREE.SphereGeometry(.145,24,14,0,Math.PI*2,0,Math.PI/2),helmet,hat);hardhat.scale.y=.9;
  const brim=cylinder(0,.004,.023,.164,.031,helmet,hat);brim.scale.z*=1.16;box(0,.117,0,.027,.017,.21,helmet,hat);
  // The helmet is a separate object so the worker can actually remove it.
  const hair=mesh(new THREE.SphereGeometry(.113,20,12,0,Math.PI*2,0,Math.PI/2),'dark',p);hair.position.y=1.245;hair.scale.y=.65;
  for(const side of [-1,1])sphere(side*.113,1.2,0,.025,.05,.047,'dark',p);
  return {root:p,hips,legs,arms,hat};
 }
 const walker=worker(.1,2.5,Math.PI/2),operator=worker(-.3,-1.05,Math.PI),supervisor=worker(2.75,1.09,-.48);
 operator.arms[0].rotation.x=-.75;operator.arms[1].rotation.x=-.8;
 supervisor.arms[0].rotation.x=-1.1;supervisor.arms[1].rotation.x=-.9;
 const tablet=box(0,.92,.27,.28,.22,.035,'dark',supervisor.root);tablet.rotation.x=-.45;
 // Proportionate surveillance camera with mount, hood, lens and power cable.
 const cctv=group(4.18,.13,-.75);box(0,.08,0,.42,.16,.42,'dark',cctv);cylinder(0,1.95,0,.075,3.8,'steel',cctv);
 bar([0,3.7,0],[-.44,3.7,0],.052,'metal',cctv);bar([0,3.18,0],[-.36,3.66,0],.032,'metal',cctv);
 const cameraPivot=group(-.43,3.72,0,cctv);cameraPivot.rotation.y=-1.2;
 const housing=roundedBox(.36,.25,.63,.06,'white',cameraPivot);housing.position.set(0,0,.18);box(0,.151,.21,.44,.045,.77,'white',cameraPivot);
 box(0,-.015,.506,.28,.2,.018,'black',cameraPivot);
 const lens=cylinder(0,-.01,.531,.086,.049,'metal',cameraPivot);lens.rotation.x=Math.PI/2;
 const glass=cylinder(0,-.01,.56,.062,.02,new THREE.MeshStandardMaterial({color:0x172936,roughness:.12,metalness:.65}),cameraPivot);glass.rotation.x=Math.PI/2;sphere(-.019,.015,.575,.016,.012,.005,0xb7cbe3,cameraPivot);
 const cameraLed=mesh(sphereGeometry,new THREE.MeshBasicMaterial({color:0x6fe5ac}),cameraPivot);cameraLed.position.set(.112,.054,.52);cameraLed.scale.set(.018,.018,.012);
 const cameraTag=label('CAM 01',.5,.17,{bg:'#242a30',color:'#fff',size:85},cctv);cameraTag.position.set(0,2.63,.089);
 // A red cone makes the camera's attention visible only while the helmet is off.
 const alarmMaterial=new THREE.MeshBasicMaterial({color:0xff1021,transparent:true,opacity:0,depthWrite:false,side:THREE.DoubleSide});
 const alarmBeam=new THREE.Mesh(new THREE.CylinderGeometry(.035,.47,1,32,1,true),alarmMaterial);world.add(alarmBeam);alarmBeam.visible=false;
 const alarmTarget=new THREE.Object3D();world.add(alarmTarget);
 const alarmLight=new THREE.SpotLight(0xff0015,0,9,.19,.7,1);alarmLight.target=alarmTarget;world.add(alarmLight);
 const groundRing=new THREE.Mesh(new THREE.RingGeometry(.40,.46,64),new THREE.MeshBasicMaterial({color:0xff2030,transparent:true,opacity:0,depthWrite:false,side:THREE.DoubleSide}));groundRing.rotation.x=-Math.PI/2;groundRing.position.y=.158;world.add(groundRing);
 const alarmGlowCanvas=document.createElement('canvas');alarmGlowCanvas.width=64;alarmGlowCanvas.height=64;const agc=alarmGlowCanvas.getContext('2d');const agg=agc.createRadialGradient(32,32,0,32,32,32);agg.addColorStop(0,'rgba(255,110,120,1)');agg.addColorStop(.18,'rgba(255,15,35,.85)');agg.addColorStop(1,'rgba(255,0,15,0)');agc.fillStyle=agg;agc.fillRect(0,0,64,64);
 const alarmGlow=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(alarmGlowCanvas),transparent:true,depthWrite:false,opacity:0,blending:THREE.AdditiveBlending}));alarmGlow.scale.set(.52,.52,1);world.add(alarmGlow);
 const alarmBadge=label('SEM CAPACETE',1.5,.3,{bg:'#bf0001',color:'#fff',size:58});alarmBadge.visible=false;
 const lensPosition=new THREE.Vector3(),pivotPosition=new THREE.Vector3(),workerTarget=new THREE.Vector3(),beamVector=new THREE.Vector3();
 const neutralCamera=new THREE.Quaternion(),aimedCamera=new THREE.Quaternion(),forwardAxis=new THREE.Vector3(0,0,1),beamAxis=new THREE.Vector3(0,-1,0),armAxis=new THREE.Vector3(0,-1,0);
 // Guard rails and bollards give the scene a believable, protected circulation.
 for(const x of [2.74,4.48]){for(const z of [-1.15,-.15])cylinder(x,.52,z,.047,.76,'yellow');bar([x,.88,-1.15],[x,.88,-.15],.045,'yellow');bar([x,.49,-1.15],[x,.49,-.15],.032,'yellow');}
 for(const x of [-4.65,4.65])for(const z of [1.95,3.12]){cylinder(x,.37,z,.08,.5,'yellow');cylinder(x,.48,z,.082,.07,'black');}
 // Fire point, hazard signage, and plants: small real-world scale cues.
 const extinguisher=cylinder(2.52,.58,-.26,.11,.67,'red');sphere(2.52,.91,-.26,.10,.07,.10,'red');bar([2.52,.99,-.26],[2.64,.99,-.26],.019,'black');
 const fireTag=label('EXTINTOR',.48,.19,{bg:'#bf0001',color:'#fff',size:72});fireTag.position.set(2.52,1.26,-.24);
 function planter(x,z){const pot=roundedBox(.65,.48,.65,.07,0x53595b);pot.position.set(x,.38,z);box(x,.63,z,.51,.018,.51,0x4c4238);for(let i=0;i<8;i++){const a=i*2.399;const leaf=sphere(x+Math.sin(a)*.16,.82+(i%3)*.09,z+Math.cos(a)*.16,.09,.32,.075,'plant');leaf.rotation.z=Math.sin(a)*.55;leaf.rotation.x=Math.cos(a)*.45;}}
 planter(-4.47,-2.83);planter(4.4,3.34);
 // Grounding shadows beneath people and equipment, like a lit physical model.
 const shadowCanvas=document.createElement('canvas');shadowCanvas.width=128;shadowCanvas.height=128;const sc=shadowCanvas.getContext('2d');const sg=sc.createRadialGradient(64,64,5,64,64,64);sg.addColorStop(0,'rgba(0,0,0,.25)');sg.addColorStop(1,'rgba(0,0,0,0)');sc.fillStyle=sg;sc.fillRect(0,0,128,128);const shadowTexture=new THREE.CanvasTexture(shadowCanvas);
 function contactShadow(parent,w,d,y){const m=new THREE.Mesh(new THREE.PlaneGeometry(w,d),new THREE.MeshBasicMaterial({map:shadowTexture,transparent:true,depthWrite:false}));m.rotation.x=-Math.PI/2;m.position.y=y;parent.add(m);}
 for(const p of [walker,operator,supervisor])contactShadow(p.root,.75,.6,-.015);contactShadow(forklift,1.9,2.3,-.015);contactShadow(conveyor,4,1.6,-.02);
 // Interaction and animation. Only the cover renders; pause survives navigation.
 const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');let paused=false,visible=!document.hidden,elapsed=0,lastTime=0,raf=0,dragging=false,pointerX=0,pointerY=0,renderCount=0;
 const gait=.18;
 let alarmActive=false,helmetOff=false,sequenceTime=0;
 const ease=n=>{n=THREE.MathUtils.clamp(n,0,1);return n*n*(3-2*n);};
 const mix=THREE.MathUtils.lerp;
 const wornHat=new THREE.Vector3(0,1.25,0),liftedHat=new THREE.Vector3(.08,1.4,.045),heldHat=new THREE.Vector3(.43,.81,.11);
 const restingHand=new THREE.Vector3(.21,.62,0),touchingHand=new THREE.Vector3(.19,1.27,.05);
 const handPosition=new THREE.Vector3(),armVector=new THREE.Vector3();
 function moveHat(progress){
  // First clear the head; then lower the helmet into the worker's hand.
  if(progress<.4)walker.hat.position.lerpVectors(wornHat,liftedHat,ease(progress/.4));
  else walker.hat.position.lerpVectors(liftedHat,heldHat,ease((progress-.4)/.6));
  walker.hat.rotation.z=-.45*ease((progress-.4)/.6);
  handPosition.copy(walker.hat.position).add(new THREE.Vector3(.15,.025,.02));
 }
 function animate(t){
  const q=t%28;sequenceTime=q;let walking=false;
  if(q<4){walker.root.position.x=mix(-1.15,.15,q/4);walker.root.rotation.y=Math.PI/2;walking=true;}
  else if(q<19){walker.root.position.x=.15;walker.root.rotation.y=q<5?mix(Math.PI/2,.35,ease(q-4)):q<18?.35:mix(.35,Math.PI/2,ease(q-18));}
  else if(q<22){walker.root.position.x=mix(.15,1.3,(q-19)/3);walker.root.rotation.y=Math.PI/2;walking=true;}
  else if(q<23){walker.root.position.x=1.3;walker.root.rotation.y=mix(Math.PI/2,-Math.PI/2,ease(q-22));}
  else if(q<27){walker.root.position.x=mix(1.3,-1.15,(q-23)/4);walker.root.rotation.y=-Math.PI/2;walking=true;}
  else{walker.root.position.x=-1.15;walker.root.rotation.y=mix(-Math.PI/2,Math.PI/2,ease(q-27));}
  const step=walking?Math.sin(t*5.8):0;
  for(let i=0;i<2;i++){walker.legs[i].rotation.x=step*(i?-1:1)*gait;walker.arms[i].rotation.set(-step*(i?-1:1)*.22,0,0);walker.arms[i].scale.y=1;}
  walker.hips.position.y=.58+(walking?Math.abs(step)*.012:0);
  walker.hat.position.copy(wornHat);walker.hat.rotation.z=0;handPosition.copy(restingHand);
  if(q>=5&&q<5.8)handPosition.lerpVectors(restingHand,touchingHand,ease((q-5)/.8));
  else if(q>=5.8&&q<6.9)moveHat((q-5.8)/1.1);
  else if(q>=6.9&&q<16)moveHat(1);
  else if(q>=16&&q<17.2)moveHat(1-(q-16)/1.2);
  else if(q>=17.2&&q<18)handPosition.lerpVectors(touchingHand,restingHand,ease((q-17.2)/.8));
  if(q>=5&&q<18){const arm=walker.arms[1];armVector.copy(handPosition).sub(arm.position);arm.quaternion.setFromUnitVectors(armAxis,armVector.clone().normalize());arm.scale.y=armVector.length()/.32;}
  helmetOff=q>=6.1&&q<17.05;
  operator.arms[0].rotation.x=-.78+Math.sin(t*.9)*.07;supervisor.root.rotation.y=-.48+Math.sin(t*.4)*.05;
  // Detection follows removal; the camera pans and tilts before the red pulses.
  const attention=ease((q-6.2)/1.0)*(1-ease((q-17.1)/1.1));
  world.updateMatrixWorld(true);cameraPivot.getWorldPosition(pivotPosition);workerTarget.set(walker.root.position.x,1.06,walker.root.position.z);
  beamVector.copy(workerTarget).sub(pivotPosition).normalize();aimedCamera.setFromUnitVectors(forwardAxis,beamVector);
  neutralCamera.setFromEuler(new THREE.Euler(0,-1.2+Math.sin(t*.35)*.16,0));cameraPivot.quaternion.copy(neutralCamera).slerp(aimedCamera,attention);
  cameraPivot.updateWorldMatrix(true,true);glass.getWorldPosition(lensPosition);lensPosition.addScaledVector(beamVector,.025);
  const alertAmount=ease((q-7.25)/.5)*(1-ease((q-17.05)/.35));alarmActive=alertAmount>.05;
  const pulse=.3+.7*(.5+.5*Math.sin((q-7.25)*Math.PI*2/1.8));
  cameraLed.material.color.setHex(alarmActive?0xff0015:0x6fe5ac);
  alarmTarget.position.copy(workerTarget);alarmLight.position.copy(lensPosition);alarmLight.intensity=alertAmount*(15+18*pulse);
  beamVector.copy(workerTarget).sub(lensPosition);alarmBeam.position.copy(lensPosition).add(workerTarget).multiplyScalar(.5);alarmBeam.quaternion.setFromUnitVectors(beamAxis,beamVector.clone().normalize());alarmBeam.scale.y=beamVector.length();alarmBeam.visible=alarmActive;alarmMaterial.opacity=alertAmount*(.045+.13*pulse);
  alarmGlow.position.copy(lensPosition);alarmGlow.material.opacity=alertAmount*(.6+.4*pulse);
  groundRing.position.x=walker.root.position.x;groundRing.position.z=walker.root.position.z;groundRing.material.opacity=alertAmount*(.3+.5*pulse);groundRing.scale.setScalar(1+pulse*.14);
  alarmBadge.visible=alarmActive;alarmBadge.position.set(walker.root.position.x,1.98,walker.root.position.z);alarmBadge.quaternion.copy(camera.quaternion);
  for(let i=0;i<parcels.length;i++){const p=(t*.17+i*1.5)%3;parcels[i].position.x=p-1.5;const fade=Math.min(1,p/.32,(3-p)/.32);parcels[i].scale.setScalar(Math.max(.01,fade));}
  for(const roll of rollers)roll.rotation.y=t*.55;
  spindle.rotation.x=t*.8;
 }
 function setCamera(){const width=panel.clientWidth,height=panel.clientHeight;if(!width||!height)return false;const aspect=width/height;const half=Math.max(5.85,7.25/aspect);camera.left=-half*aspect;camera.right=half*aspect;camera.top=half;camera.bottom=-half;camera.updateProjectionMatrix();camera.position.set(Math.sin(yaw)*18*Math.cos(pitch),Math.sin(pitch)*18,Math.cos(yaw)*18*Math.cos(pitch));camera.position.add(target);camera.lookAt(target);return true;}
 function renderStill(){if(!renderer||!camera||!panel.clientWidth||!panel.clientHeight)return;setCamera();renderer.render(scene,camera);renderCount++;}
 function resize(){const width=panel.clientWidth,height=panel.clientHeight;if(!width||!height)return;renderer.setSize(width,height,false);renderStill();}
 function canAnimate(){return cover.classList.contains('active')&&visible&&!paused&&!reduced.matches;}
 function tick(now){raf=0;if(!canAnimate()){lastTime=0;return;}if(!lastTime)lastTime=now;const dt=now-lastTime;if(dt>=30){elapsed+=Math.min(dt,100)/1000;lastTime=now;yaw+=(desiredYaw-yaw)*.12;pitch+=(desiredPitch-pitch)*.12;animate(elapsed);renderStill();}raf=requestAnimationFrame(tick);}
 function sync(){cancelAnimationFrame(raf);raf=0;lastTime=0;resize();if(canAnimate())raf=requestAnimationFrame(tick);}
 canvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;dragging=true;pointerX=e.clientX;pointerY=e.clientY;canvas.setPointerCapture(e.pointerId);});
 canvas.addEventListener('pointermove',e=>{if(!dragging)return;desiredYaw=Math.max(-.5,Math.min(1.4,desiredYaw+(e.clientX-pointerX)*.005));desiredPitch=Math.max(.35,Math.min(.9,desiredPitch+(e.clientY-pointerY)*.003));pointerX=e.clientX;pointerY=e.clientY;yaw=desiredYaw;pitch=desiredPitch;renderStill();});
 for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>{dragging=false;});
 canvas.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home'].includes(e.key))return;e.preventDefault();e.stopPropagation();if(e.key==='Home'){desiredYaw=.73;desiredPitch=.60;}else if(e.key==='ArrowLeft')desiredYaw=Math.max(-.5,desiredYaw-.12);else if(e.key==='ArrowRight')desiredYaw=Math.min(1.4,desiredYaw+.12);else if(e.key==='ArrowUp')desiredPitch=Math.min(.9,desiredPitch+.07);else desiredPitch=Math.max(.35,desiredPitch-.07);yaw=desiredYaw;pitch=desiredPitch;renderStill();});
 resetButton.onclick=()=>{yaw=desiredYaw=.73;pitch=desiredPitch=.60;renderStill();};
 pauseButton.onclick=()=>{paused=!paused;pauseButton.setAttribute('aria-pressed',String(paused));pauseButton.setAttribute('aria-label',paused?'Retomar maquete':'Pausar maquete');pauseButton.title=paused?'Retomar maquete':'Pausar maquete';document.getElementById('scene-pause-icon').innerHTML=paused?'<path d="m6 3 8 6-8 6Z"/>':'<path d="M6 3v12M12 3v12"/>';sync();};
 new ResizeObserver(resize).observe(panel);new MutationObserver(sync).observe(cover,{attributes:true,attributeFilter:['class']});document.addEventListener('visibilitychange',()=>{visible=!document.hidden;sync();});reduced.addEventListener('change',sync);
 canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();cancelAnimationFrame(raf);panel.classList.add('no-webgl');pauseButton.hidden=true;resetButton.hidden=true;});
 canvas.addEventListener('webglcontextrestored',()=>{panel.classList.remove('no-webgl');pauseButton.hidden=false;resetButton.hidden=false;sync();});
 window.addEventListener('beforeprint',()=>{renderStill();document.getElementById('diorama-fallback').src=canvas.toDataURL('image/png');});
 // Read-only diagnostics for this scene; useful when checking the cover locally.
 window.epiDioramaState=()=>({ready:true,paused,active:cover.classList.contains('active'),reducedMotion:reduced.matches,frames:renderCount,elapsed:Number(elapsed.toFixed(2)),sequenceTime:Number(sequenceTime.toFixed(2)),helmetOff,alarmActive,yaw,pitch,meshes:renderer.info.render.calls,width:canvas.width,height:canvas.height});
 loading.hidden=true;animate(2);resize();officialLogo.addEventListener('load',paintLogo);paintLogo();sync();
})();
