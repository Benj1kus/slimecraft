import { Engine } from 'noa-engine'
import { CreateBox } from '@babylonjs/core/Meshes/Builders/boxBuilder'
import { HemisphericLight } from '@babylonjs/core/Lights/hemisphericLight'
import { Vector3 } from '@babylonjs/core/Maths/math.vector'
import { Color3 } from '@babylonjs/core/Maths/math.color'

const noa = new Engine({ 
    debug: true, 
    showFPS: true, 
    chunkSize: 32,
    chunkAddDistance: 2.5,
    chunkRemoveDistance: 3.5 
})

const AI_SERVER = "http://127.0.0.1:8000"
const scene = noa.rendering.getScene()

// ====================
// 1. Освещение
// ====================
const light = new HemisphericLight("light", new Vector3(0,1,0), scene)
light.intensity = 1.0
scene.ambientColor = new Color3(1,1,1)

// ====================
// 2. Биомы
// ====================
const PLANET_TYPES = ['tropical','volcanic','crystal','toxic','frozen','desert','nebular','fungal','aquatic','metallic','lush','artic','synthetic']
const selectedBiomes = PLANET_TYPES.sort(() => 0.5 - Math.random()).slice(0,3)
console.log("--- СЕССИЯ: " + selectedBiomes.join(", ") + " ---")

// ====================
// 3. Материалы и блоки
// ====================
const time = Date.now()
const types = ['ground','bark','leaves']

const biomeBlocks = {}
selectedBiomes.forEach((biome, index) => {
    biomeBlocks[biome] = {}

    types.forEach(t => {
        const matName = `${biome}_${t}`
        noa.registry.registerMaterial(matName, {
            textureURL: `${AI_SERVER}/static/${biome}/${t}_albedo.png?v=${time}`
        })
    })

    biomeBlocks[biome].ground = noa.registry.registerBlock(index*10+1, { material: `${biome}_ground` })
    biomeBlocks[biome].bark   = noa.registry.registerBlock(index*10+2, { material: `${biome}_bark` })
    biomeBlocks[biome].leaves = noa.registry.registerBlock(index*10+3, { material: `${biome}_leaves` })
})

// --- ИСПРАВЛЕННЫЙ DIRT: ТЕПЕРЬ БЕРЕТ ТЕКСТУРУ ИЗ CVAE ---
noa.registry.registerMaterial('dirt_mat', { 
    textureURL: `${AI_SERVER}/static/dirt/ground_albedo.png?v=${time}` 
})
biomeBlocks['dirt'] = { id: noa.registry.registerBlock(999, { material: 'dirt_mat' }) }

// ====================
// 3a. Вода
// ====================
noa.registry.registerMaterial('water_mat', { color: [0.1, 0.4, 0.8], alpha: 0.6 })
const waterBlock = noa.registry.registerBlock(1000, { 
    material: 'water_mat', 
    opaque: false,
    fluid: true,
    solid: false 
})

// ====================
// 4. Плавный noise
// ====================
function smoothNoise(x, z) {
    return (Math.sin(x*0.01) + Math.cos(z*0.01)) / 2 + 0.5
}

function chooseBiome(x, z) {
    const n = smoothNoise(x, z)
    const index = Math.floor(n * selectedBiomes.length)
    return selectedBiomes[index % selectedBiomes.length]
}

// ====================
// 5. Деревья
// ====================
function getTreeVoxel(x, y, z, baseHeight, blocks, biome) {
    const waterLevel = 3
    if (y <= waterLevel) return 0

    const config = {
        tropical: { chance: 0.980, minH: 6, maxH: 12, shape: 'cone' },
        frozen:   { chance: 0.994, minH: 3, maxH: 5,  shape: 'cube' },
        crystal:  { chance: 0.988, minH: 5, maxH: 9,  shape: 'sphere' },
        default:  { chance: 0.992, minH: 4, maxH: 7,  shape: 'sphere' }
    }
    const p = config[biome] || config.default

    const getHash = (nx, nz) => (Math.abs(Math.sin(nx * 12.9898 + nz * 78.233) * 43758.5453) % 1)
    const radius = 3
    for (let dx = -radius; dx <= radius; dx++) {
        for (let dz = -radius; dz <= radius; dz++) {
            const nx = x + dx
            const nz = z + dz
            const nHash = getHash(nx, nz)
            if (nHash > p.chance) {
                const treeSeed = (nHash * 1000) % 1
                const currentTreeHeight = Math.floor(p.minH + treeSeed * (p.maxH - p.minH))
                const currentCrownSize = Math.floor(2 + treeSeed * 2)
                const relY = y - baseHeight
                const distXZ = Math.sqrt(dx * dx + dz * dz)
                
                if (dx === 0 && dz === 0 && relY >= 0 && relY < currentTreeHeight) return blocks.bark
                
                const crownStartY = currentTreeHeight - 1
                if (relY >= crownStartY && relY < currentTreeHeight + currentCrownSize + 1) {
                    const localY = relY - crownStartY
                    switch(p.shape) {
                        case 'cube':
                            if (Math.abs(dx) <= 1 && Math.abs(dz) <= 1) return blocks.leaves
                            break
                        case 'sphere':
                            if (distXZ + Math.abs(localY - currentCrownSize/2) <= currentCrownSize * 0.7) return blocks.leaves
                            break
                        case 'cone':
                            const coneWidth = currentCrownSize * (1 - localY / (currentCrownSize + 2))
                            if (distXZ <= coneWidth) return blocks.leaves
                            break
                    }
                }
            }
        }
    }
    return 0
}

// ====================
// 6. Высота рельефа и слои (Ground -> Dirt)
// ====================
function getVoxelID(x, y, z) {
    const biome = chooseBiome(x, z)
    const blocks = biomeBlocks[biome]
    const dirtID = biomeBlocks['dirt'].id

    let baseHeight = 5
    switch(biome){
        case 'tropical': baseHeight += Math.floor(Math.sin(x*0.05)*4 + Math.cos(z*0.05)*4); break
        case 'volcanic': baseHeight += Math.floor(Math.sin(x*0.1)*5); break
        case 'crystal':  baseHeight += Math.floor(Math.sin(x*0.06)*8); break
        case 'frozen':   baseHeight += 2; break 
        default:         baseHeight += Math.floor(Math.sin(x*0.05)*2 + Math.cos(z*0.05)*2); break
    }

    const waterLevel = 3

    if (y < -15) return dirtID

    if (y < baseHeight) {
        // Поверхностный слой (трава биома)
        if (y >= baseHeight - 1) return blocks.ground
        // Всё что ниже — Dirt (теперь с текстурой)
        return dirtID
    }

    if (y <= waterLevel) return waterBlock

    if (baseHeight > waterLevel) {
        return getTreeVoxel(x, y, z, baseHeight, blocks, biome)
    }

    return 0
}

// ====================
// 7. Генерация чанков
// ====================
noa.world.on('worldDataNeeded', (id, data, x, y, z) => {
    for (let i=0;i<data.shape[0];i++)
        for (let j=0;j<data.shape[1];j++)
            for (let k=0;k<data.shape[2];k++){
                data.set(i,j,k, getVoxelID(x+i, y+j, z+k))
            }
    noa.world.setChunkData(id, data)
})

// ====================
// 8. Игрок
// ====================
const player = noa.playerEntity
noa.entities.setPosition(player, [0, 20, 0]) 
const dat = noa.entities.getPositionData(player)
const pMesh = CreateBox('player-mesh', {}, scene)
pMesh.scaling.set(dat.width, dat.height, dat.width)
noa.entities.addComponent(player, noa.entities.names.mesh, { mesh: pMesh, offset: [0, dat.height/2, 0] })

// ====================
// 9. Взаимодействие
// ====================
noa.inputs.down.on('fire', () => {
    if (noa.targetedBlock) noa.setBlock(0, ...noa.targetedBlock.position)
})

noa.inputs.down.on('alt-fire', () => {
    if (noa.targetedBlock) {
        const pos = noa.targetedBlock.adjacent
        const biome = chooseBiome(pos[0], pos[2])
        noa.setBlock(biomeBlocks[biome].ground, ...pos)
    }
})

noa.inputs.bind('alt-fire', 'KeyE')

console.log("🌍 Текстурированный мир готов: Грязь берется из CVAE!")